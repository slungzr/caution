import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    filename=f'C:\\wencai\\竞价爬升统计_{current_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def analyze_with_threshold(df, threshold, base_output_path):
    logging.info(f"开始分析 threshold={threshold} 的数据")
    
    # 过滤数据
    filtered_df = df[df['个股热度排名昨日'] <= threshold].copy()
    filtered_df['真实日期'] = filtered_df['日期']
    
    # 计算收盘价收益率
    filtered_df['收盘收益率'] = (filtered_df['收盘价:不复权明日'].astype(float) - 
                                filtered_df['收盘价:不复权今日'].astype(float)) / \
                                filtered_df['收盘价:不复权今日'].astype(float)

    # 生成包含股票简称和6位股票代码的新列，格式为 名称(300111)
    import re
    # 尝试检测可能的代码列名，优先使用存在的代码列
    code_cols = [c for c in filtered_df.columns if any(k in c for k in ['代码', '证券代码', '股票代码'])]
    code_col = code_cols[0] if code_cols else None

    def format_code_from_value(v):
        s = str(v)
        # 提取所有连续数字并合并
        digits = re.findall(r"\d+", s)
        if not digits:
            return ''
        combined = ''.join(digits)
        # 保留最后6位并左侧补0到6位
        return combined[-6:].zfill(6)

    if code_col:
        filtered_df['股票简称_代码'] = filtered_df['股票简称'].astype(str) + '(' + filtered_df[code_col].apply(format_code_from_value) + ')'
    else:
        # 从股票简称中尝试提取代码（比如 名称(300111) 或 名称300111）
        def build_name_code(name):
            name_str = str(name)
            code = format_code_from_value(name_str)
            if code:
                # 去掉原有括号内的内容以避免重复
                name_only = re.sub(r"\(.*?\)", '', name_str).strip()
                return f"{name_only}({code})"
            else:
                return name_str
        filtered_df['股票简称_代码'] = filtered_df['股票简称'].apply(build_name_code)

    # 将 '股票简称_代码' 拆成两列：'名称' 和 '代码'（6位），并保留原列
    def split_name_code(s):
        s = str(s)
        # 若形如 名称(300111)
        m = re.match(r"^(.*?)\((\d{1,})\)$", s)
        if m:
            name = m.group(1).strip()
            code = m.group(2)[-6:].zfill(6)
            return pd.Series({'名称': name, '代码': code})
        # 否则尝试提取尾部数字
        digits = re.findall(r"(\d+)", s)
        if digits:
            code = digits[-1][-6:].zfill(6)
            name = re.sub(r"\(.*?\)", '', s).strip()
            name = re.sub(r"\d+", '', name).strip()
            return pd.Series({'名称': name, '代码': code})
        return pd.Series({'名称': s, '代码': ''})

    name_code_df = filtered_df['股票简称_代码'].apply(split_name_code)
    # 将拆分得到的两列合并回 filtered_df（若已存在同名列则覆盖）
    filtered_df['名称'] = name_code_df['名称']
    filtered_df['代码'] = name_code_df['代码']

    # 保存每个阈值的过滤后明细数据到CSV，包含新的列
    filtered_threshold_file = f"{base_output_path}-过滤后数据-{threshold}.csv"
    try:
        # 准备一份用于保存的副本，不改变原始 filtered_df 的数据类型
        save_df = filtered_df.copy()
        # 确保 '代码' 是6位字符串
        save_df['代码'] = save_df['代码'].astype(str).str.zfill(6)
        # 为了在 Excel 中保留前导0，将代码格式化为 ="000123" 形式（Excel 会显示为 000123）
        save_df['代码'] = save_df['代码'].apply(lambda x: f'="{x}"' if x and x.strip() != "" else '')

        # 把代码放在最前，名称其次，随后添加真实日期列，其他列保留原始顺序
        front = ['代码', '名称', '真实日期']
        cols = [c for c in list(save_df.columns) if c not in front]
        save_cols = [c for c in front if c in save_df.columns] + cols
        save_df = save_df[save_cols]

        save_df.to_csv(filtered_threshold_file, encoding='utf-8-sig', index=False)
        logging.info(f"阈值 {threshold} 的过滤后明细已保存到: {filtered_threshold_file}")
    except Exception as e:
        logging.exception(f"保存阈值 {threshold} 的过滤后明细时出错: {e}")

    # 按日期分组计算统计数据，使用新列 '股票简称_代码'
    stats = filtered_df.groupby('真实日期').agg({
        '开盘收益率': ['count', 'mean', 'std', 'min', 'max'],
        '收益率': ['mean', 'std', 'min', 'max'],
        '收盘收益率': ['mean', 'std', 'min', 'max'],
        '股票简称_代码': lambda x: ','.join(x)
    }).round(4)
    
    # 重命名列
    stats.columns = [
        '股票数量',
        '开盘收益率均值',
        '开盘收益率标准差',
        '开盘收益率最小值',
        '开盘收益率最大值',
        '收益率均值',
        '收益率标准差',
        '收益率最小值',
        '收益率最大值',
        '收盘收益率均值',
        '收盘收益率标准差',
        '收盘收益率最小值',
        '收盘收益率最大值',
        '股票简称'
    ]
    
    # 转换日期格式并排序
    stats.index = pd.to_datetime(stats.index, format='%Y%m%d')
    stats = stats.sort_index(ascending=True)
    
    # 计算收益率复利
    stats['复利'] = (1 + stats['收益率均值']).cumprod() - 1
    
    # 将日期格式转回原来的格式
    stats.index = stats.index.strftime('%Y%m%d')
    
    # 保存日统计结果
    daily_output_file = f'{base_output_path}-统计-{threshold}.csv'
    stats.to_csv(daily_output_file, encoding='utf-8-sig')
    logging.info(f"阈值 {threshold} 的日统计结果已保存到: {daily_output_file}")
    
    # 计算月度统计
    stats['月份'] = pd.to_datetime(stats.index, format='%Y%m%d').strftime('%Y-%m')
    monthly_stats = stats.groupby('月份').agg({
        '股票数量': 'sum',
        '开盘收益率均值': 'sum',
        '收益率均值': 'sum',
        '收盘收益率均值': 'sum',
        '复利': 'last'
    }).round(4)
    
    # 保存月度统计结果
    monthly_output_file = f'{base_output_path}-月度统计-{threshold}.csv'
    monthly_stats.to_csv(monthly_output_file, encoding='utf-8-sig')
    logging.info(f"阈值 {threshold} 的月度统计结果已保存到: {monthly_output_file}")
    
    return stats, monthly_stats

def main():
    # 读取CSV文件
    csv_file = 'C:\\wencai\\竞价爬升-20240504.csv'
    base_output_path = csv_file.replace('.csv', '')
    
    logging.info(f"开始读取CSV文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file, encoding='gb18030')
    except UnicodeDecodeError:
        logging.warning("GB18030编码失败，尝试UTF-8编码")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    logging.info("CSV文件读取成功")
    
    # 确保日期列是字符串类型
    df['日期'] = df['日期'].astype(str)
    # Add day of week column (0=Monday, 6=Sunday)
    df['星期几'] = pd.to_datetime(df['日期'], format='%Y%m%d').dt.dayofweek + 1
    # Print data types of all columns
    print("\nColumn data types:")
    print(df.dtypes)
    logging.info("\nColumn data types:")
    logging.info(df.dtypes)
    # 过滤实体涨跌幅昨日大于等于0的数据

    df = df[df['实体涨跌幅昨日'] <df['实体涨跌幅前日'] ] 
    # 过滤前日成交量大于昨日成交量的数据
    df = df[(df['成交量前日'] < df['成交量昨日']) ]
    # df = df[(df['成交量前日'] < df['成交量昨日']) & (df['实体涨跌幅昨日'] == 0)]
    logging.info(f"过滤后剩余数据条数: {len(df)}")
    
    # 计算收盘收益率
    df['收盘收益率'] = (df['收盘价:不复权明日'].astype(float) - 
                        df['收盘价:不复权今日'].astype(float)) / \
                        df['收盘价:不复权今日'].astype(float)
    
    # 保存过滤后的数据到CSV文件
    filtered_output_file = f'{base_output_path}-过滤后数据.csv'
# 按照日期升序排列
    df = df.sort_values(by='日期')
    df.to_csv(filtered_output_file, encoding='utf-8-sig', index=False)
    logging.info(f"过滤后的数据已保存到: {filtered_output_file}")
    
    # 定义要分析的阈值列表
    thresholds = [10, 15,20,25, 30, 50, 100]
    
    # 存储所有结果用于比较
    all_results = {}
    
    # 对每个阈值进行分析
    for threshold in thresholds:
        logging.info(f"\n开始分析阈值 {threshold}")
        daily_stats, monthly_stats = analyze_with_threshold(df, threshold, base_output_path)
        all_results[threshold] = {
            'daily': daily_stats,
            'monthly': monthly_stats
        }
        
        print(f"\n阈值 {threshold} 的统计结果:")
        print("\n日统计概览:")
        print(daily_stats)
        print("\n月度统计概览:")
        print(monthly_stats)
    
    # 创建比较报告
    comparison_data = {
        '最终复利': {},
        '平均每月收益率': {}
    }
    
    for threshold in thresholds:
        comparison_data['最终复利'][threshold] = all_results[threshold]['monthly']['复利'].iloc[-1]
        comparison_data['平均每月收益率'][threshold] = all_results[threshold]['monthly']['收益率均值'].mean()
    
    comparison_df = pd.DataFrame(comparison_data).round(4)
    
    comparison_output_file = f'{base_output_path}-阈值比较.csv'
    comparison_df.to_csv(comparison_output_file, encoding='utf-8-sig')
    logging.info(f"\n阈值比较结果:\n{comparison_df}")
    print(f"\n阈值比较结果已保存到: {comparison_output_file}")
    print(comparison_df)

    # 绘制复利曲线图
    plt.figure(figsize=(15, 8))
    # 色弱/色盲友好的调色板（8色），并配合不同线型和标记以提高可辨识度
    cb_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'x']
    for i, threshold in enumerate(thresholds):
        daily_stats = all_results[threshold]['daily']
        color = cb_palette[i % len(cb_palette)]
        ls = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        plt.plot(pd.to_datetime(daily_stats.index),
                 daily_stats['复利'],
                 label=f'热度阈值 {threshold}',
                 color=color,
                 linestyle=ls,
                 marker=marker,
                 markevery=max(1, len(daily_stats)//10),
                 linewidth=2,
                 markersize=5)

    plt.title('不同热度阈值的复利曲线对比')
    plt.xlabel('日期')
    plt.ylabel('复利')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    
    # 保存图表
    plt.tight_layout()  # 自动调整布局
    plot_output_file = f'{base_output_path}-复利曲线.png'
    plt.savefig(plot_output_file, dpi=300, bbox_inches='tight')
    logging.info(f"复利曲线图已保存到: {plot_output_file}")
    plt.close()

if __name__ == "__main__":
    main()