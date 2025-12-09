import os
import re
import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict

def natural_sort_key(key):
    """
    自然な順序でソートするためのキー生成関数
    例: layers.9 < layers.10 となるようにする
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    
    return alphanum_key(key)

def sort_state_dict(state_dict):
    """
    state_dictのキーを自然な順序でソート
    """
    sorted_keys = sorted(state_dict.keys(), key=natural_sort_key)
    sorted_dict = OrderedDict()
    for key in sorted_keys:
        sorted_dict[key] = state_dict[key]
    return sorted_dict

def get_tensor_size(tensor):
    """テンソルのサイズをバイト単位で取得"""
    return tensor.element_size() * tensor.nelement()

def save_split_safetensors(state_dict, output_path, max_size_gb=4, model_name="model"):
    """
    state_dictを指定サイズごとに分割してsafetensors形式で保存
    
    Args:
        state_dict: 保存する重みの辞書
        output_path: 出力ディレクトリパス
        max_size_gb: 各ファイルの最大サイズ（GB単位）
        model_name: モデル名（デフォルト: "model"）
    """
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # GBをバイトに変換
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_path, exist_ok=True)
    
    # state_dictを自然な順序でソート
    print("\n重みを自然な順序でソートしています...")
    sorted_state_dict = sort_state_dict(state_dict)
    
    current_shard = OrderedDict()
    current_size = 0
    shard_index = 1
    saved_files = []
    total_shards = 1  # 総シャード数を推定
    
    # 総シャード数を事前に計算
    temp_size = 0
    for tensor in sorted_state_dict.values():
        tensor_size = get_tensor_size(tensor)
        if temp_size + tensor_size > max_size_bytes and temp_size > 0:
            total_shards += 1
            temp_size = tensor_size
        else:
            temp_size += tensor_size
    
    # インデックスファイル用の情報
    weight_map = {}
    
    print(f"\n重みを最大{max_size_gb}GBごとに分割して保存します...")
    print(f"推定シャード数: {total_shards}")
    
    for key, tensor in sorted_state_dict.items():
        tensor_size = get_tensor_size(tensor)
        
        # 現在のシャードに追加すると制限を超える場合、現在のシャードを保存
        if current_size + tensor_size > max_size_bytes and current_shard:
            # ファイル名を生成（model名を使用）
            shard_filename = f"{model_name}-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            shard_path = os.path.join(output_path, shard_filename)
            
            # safetensors形式で保存
            print(f"保存中: {shard_filename} (サイズ: {current_size / (1024**3):.2f} GB, キー数: {len(current_shard)})")
            save_file(current_shard, shard_path)
            saved_files.append(shard_filename)
            
            # weight_mapを更新
            for k in current_shard.keys():
                weight_map[k] = shard_filename
            
            # 次のシャードの準備
            current_shard = OrderedDict()
            current_size = 0
            shard_index += 1
        
        # テンソルを現在のシャードに追加
        current_shard[key] = tensor
        current_size += tensor_size
    
    # 最後のシャードを保存
    if current_shard:
        shard_filename = f"{model_name}-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(output_path, shard_filename)
        
        print(f"保存中: {shard_filename} (サイズ: {current_size / (1024**3):.2f} GB, キー数: {len(current_shard)})")
        save_file(current_shard, shard_path)
        saved_files.append(shard_filename)
        
        # weight_mapを更新
        for k in current_shard.keys():
            weight_map[k] = shard_filename
    
    # インデックスファイルを作成
    index_file = {
        "metadata": {
            "total_size": sum(get_tensor_size(t) for t in sorted_state_dict.values()),
            "format": "safetensors"
        },
        "weight_map": weight_map
    }
    
    import json
    index_path = os.path.join(output_path, f"{model_name}.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index_file, f, indent=2)
    
    print(f"\nインデックスファイル保存: {index_path}")
    print(f"合計 {len(saved_files)} 個のファイルに分割して保存完了")
    
    return saved_files

def convert_weight_names(state_dict):
    """
    重みの名前を変換する関数
    （主に safetensors 側の変換用）
    """
    name_mapping = {
        'dummydummy.': '',
    }
    
    new_state_dict = OrderedDict()
    
    with torch.no_grad():
        for old_key, tensor in state_dict.items():
            new_key = old_key
            for old_pattern, new_pattern in name_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            # テンソルをBF16に変換し、grad情報なしでコピー
            print(f'old = {old_key} new = {new_key}')
            new_state_dict[new_key] = tensor.detach().clone().to(torch.bfloat16)
    
    return new_state_dict

def convert_adapter_weight_names(state_dict):
    """
    重みの名前を変換する関数（Adapter.bin 用）
    例: model.model.layers.23.self_attn.student_attn.v0 -> layers.23.att.v0
    """
    adapter_name_mapping = {
        'model.model.': 'model.',
        'self_attn.student_attn.': 'self_attn.',
    }
    
    new_state_dict = OrderedDict()
    
    with torch.no_grad():
        for old_key, tensor in state_dict.items():
            print(f'Adapter File {old_key} {tensor.shape}')
            new_key = old_key
            for old_pattern, new_pattern in adapter_name_mapping.items():
                new_key = new_key.replace(old_pattern, new_pattern)
            new_state_dict[new_key] = tensor.detach().clone().to(torch.bfloat16)
    
    return new_state_dict

def merge_safetensors(input_dir):
    """
    指定されたディレクトリ内のすべてのsafetensorファイルを読み込み、キー名変換後にマージする
    """
    safetensor_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]
    
    if not safetensor_files:
        raise FileNotFoundError("Inputフォルダにsafetensorsファイルが見つかりません。")
    
    # ファイル名を自然な順序でソート
    safetensor_files.sort(key=natural_sort_key)
    
    print(f"処理するファイル数: {len(safetensor_files)}")
    
    merged_weights = OrderedDict()
    
    for file_name in safetensor_files:
        safetensor_path = os.path.join(input_dir, file_name)
        print(f"\n処理中のファイル: {file_name}")
        
        try:
            with torch.no_grad():
                # safetensorsファイルを読み込み
                current_weights = load_file(safetensor_path)
                
                # 重み名を変換
                converted_weights = convert_weight_names(current_weights)
                
                # 重複をチェック
                overlap = set(merged_weights.keys()) & set(converted_weights.keys())
                if overlap:
                    print(f"\n警告: {file_name} で重複する重みが見つかりました:")
                    for key in overlap:
                        print(f"  {key}")
                    choice = input("重複する重みを上書きしますか？ (y/n): ").strip().lower()
                    if choice != 'y':
                        print("処理を中断します。")
                        return None
                
                # 重みをマージ（BF16形式、grad情報なし）
                for key, tensor in converted_weights.items():
                    print(f'safetensor {key}')
                    merged_weights[key] = tensor.clone()
                    
                print(f"{file_name} の処理が完了しました。")
                print(f"現在の総重み数: {len(merged_weights)}")
            
        except Exception as e:
            print(f"ファイル {file_name} の処理中にエラーが発生しました: {str(e)}")
            return None
    
    return merged_weights

import sys

def main():
    if len(sys.argv) != 4:
        print(f"Converts from HF + partial PTH from stage 2 training to safetensors format")
        print("Usage: python ConvertToRWKVInfer_hybrid_safetensors.py teacher_hf_path in_pth_file out_path")
        exit()

    input_dir = sys.argv[1]
    adapter_file = sys.argv[2]
    output_path = sys.argv[3]
    
    model_name = "model"  # モデル名
    max_size_gb = 4  # 各ファイルの最大サイズ（GB）
    
    try:
        print("safetensorファイルのマージを開始します...")
        with torch.no_grad():
            merged_weights = merge_safetensors(input_dir)
            
            # ---- Adapter.bin を読み込み、キー変換してマージ ----
            print(f"\nPyTorchモデルファイル {adapter_file} を読み込みます...")
            
            if not os.path.isfile(adapter_file):
                raise FileNotFoundError(f"Adapterファイルが見つかりません: {adapter_file}")
            
            adapter_state = torch.load(adapter_file, map_location="cpu")
            print(f"Adapterファイル読み込み完了。キー数: {len(adapter_state)}")
            
            print("Adapter用のキー変換を実行します...")
            converted_adapter_weights = convert_adapter_weight_names(adapter_state)
            
            # Check which layers have RWKV (by detecting receptance)
            rwkv_layers = set()
            for i in range(96):
                SearchWord = f'layers.{i}.self_attn.receptance'
                for key in converted_adapter_weights.keys():
                    if SearchWord in key:
                        rwkv_layers.add(i)

            print(converted_adapter_weights.keys())
           # exit()
            
            print(f'検出されたRWKVレイヤー: {sorted(rwkv_layers)}')
            print(f'RWKVレイヤー数: {len(rwkv_layers)}')
            
            # Get total number of layers
            TotalLayers = 0
            for i in range(96):
                SearchWord = f'layers.{i}.'
                for key in merged_weights.keys():
                    if SearchWord in key:
                        TotalLayers = i + 1
            print(f'TotalLayers = {TotalLayers}')
            
            # Count GQA layers (layers without RWKV)
            gqa_layers = set(range(TotalLayers)) - rwkv_layers
            print(f'GQAレイヤー: {sorted(gqa_layers)}')
            print(f'GQAレイヤー数: {len(gqa_layers)}')

            #exit()
            
            # Remove q_proj, k_proj, v_proj, o_proj, qkv_proj only from RWKV layers
            remove_keys = []
            for i in rwkv_layers:
                att = f'layers.{i}.self_attn'
                for k in merged_weights.keys():
                    if att in k and ("r_proj" in k or "q_proj" in k or "k_proj" in k or "v_proj" in k or "o_proj" in k or "qkv_proj" in k or "q_norm" in k):
                        remove_keys.append(k)
            
            if merged_weights is None:
                print("マージ処理が中断されました。")
                return
            
            # q_proj, k_proj, v_proj を含むキーを削除（RWKVレイヤーのみ）
            print(f"\nRWKVレイヤー {sorted(rwkv_layers)} からq_proj, k_proj, v_proj等を削除します...")
            if remove_keys:
                print(f"削除対象のキー:")
                for k in sorted(remove_keys, key=natural_sort_key):
                    print(f"  - {k}")
            else:
                print("削除対象のキーはありません。")
           # exit()
            for k in remove_keys:
                del merged_weights[k]
                print(f'{k} deleted')
            print(f"削除したキー数: {len(remove_keys)}")
            
            # マージ時の重複チェック
            overlap = set(merged_weights.keys()) & set(converted_adapter_weights.keys())
            if overlap:
                print(f"\n警告: Adapter と既存のモデルで重複するキーが見つかりました:")
                for key in overlap:
                    print(f"  {key}")
                choice = input("重複するキーを上書きしますか？ (y/n): ").strip().lower()
                if choice != 'y':
                    print("処理を中断します。")
                    return
            
            # Adapter の重みをマージ
            for key, tensor in converted_adapter_weights.items():
                merged_weights[key] = tensor.clone()
            
            print(f"Adapter のマージが完了しました。現在の総重み数: {len(merged_weights)}")
            
            have_head = False
            for key, tensor in merged_weights.items():
                print(f'merged key = {key} {tensor.shape}')
                if 'head' in key:
                    print(f'have head {key}')
                    have_head = True
            
            # if have_head == False:
            #     merged_weights['head.weight'] = merged_weights['emb.weight'].clone()
            
            # ---- safetensors形式で4GBごとに分割して保存 ----
            print(f"\nマージした重みを分割して保存します...")
            saved_files = save_split_safetensors(
                merged_weights, 
                output_path, 
                max_size_gb=max_size_gb,
                model_name=model_name
            )
            
            print(f"\n保存完了！")
            print(f"出力ディレクトリ: {output_path}")
            print(f"保存されたファイル:")
            for file in saved_files:
                file_path = os.path.join(output_path, file)
                file_size = os.path.getsize(file_path) / (1024**3)
                print(f"  - {file} ({file_size:.2f} GB)")
            
            # ソート済みの重み名一覧を表示（最初の20個）
            print("\n最終的な重み名一覧（ソート済み）:")
            sorted_keys = sorted(merged_weights.keys(), key=natural_sort_key)
            for i, key in enumerate(sorted_keys[:20]):
                print(f"  {i+1:3d}. {key}")
            if len(sorted_keys) > 20:
                print(f"  ... (合計 {len(merged_weights)} 個の重み)")
            
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()