import json
import os
import argparse


def merge_json_files(outputs_dir):
    # 初始化一个空列表来存储所有 JSON 文件的内容
    merged_data = []

    # 遍历 outputs_dir 下的所有子目录
    for entry in os.scandir(outputs_dir):
        if entry.is_dir():  # 只处理子目录
            subdir = entry.path
            # 遍历子目录中的文件
            for filename in os.listdir(subdir):
                if filename.endswith(".json"):
                    file_path = os.path.join(subdir, filename)
                    with open(file_path, "r") as file:
                        # 读取 JSON 文件内容并添加到 merged_data 列表中
                        data = json.load(file)
                        merged_data.append(data)

    # 将合并后的数据写入一个新的 JSON 文件
    merged_output_path = os.path.join(outputs_dir, "results.json")
    with open(merged_output_path, "w") as outfile:
        json.dump(merged_data, outfile, indent=4)

    print(f"JSON 文件合并完成，结果保存在 {merged_output_path} 中。")


if __name__ == "__main__":
    # 设置 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="合并指定目录中的所有 JSON 文件")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="包含 JSON 文件的目录路径"
    )
    args = parser.parse_args()

    # 调用合并函数
    merge_json_files(args.output_dir)
