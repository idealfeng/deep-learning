import json

# 输入文件（你的原始数据集）
input_filepath = "./girl.jsonl"
# 输出文件（一个全新的、干净的数据集）
output_filepath = "./girl_cleaned.jsonl"

valid_records = []
line_number = 0

print(f"开始清洗数据文件: {input_filepath} ...")

with open(input_filepath, 'r', encoding='utf-8') as infile:
    for line in infile:
        line_number += 1
        try:
            # 尝试加载每一行作为JSON对象
            record = json.loads(line)

            # 【核心检查】确保 instruction 和 output 都存在且不是None
            if 'instruction' in record and 'output' in record and \
                    record['instruction'] is not None and record['output'] is not None:
                valid_records.append(line)
            else:
                print(f"警告：第 {line_number} 行数据无效，将被丢弃。内容: {line.strip()}")

        except json.JSONDecodeError:
            print(f"错误：第 {line_number} 行不是有效的JSON格式，将被丢弃。内容: {line.strip()}")

print(f"\n清洗完成！共发现 {len(valid_records)} 条有效数据。")

# 将所有有效的数据写回到一个新的文件里
with open(output_filepath, 'w', encoding='utf-8') as outfile:
    for record_line in valid_records:
        outfile.write(record_line)

print(f"已将干净的数据保存到: {output_filepath}")
print("请修改你的训练脚本，使用这个新的干净文件进行训练。")