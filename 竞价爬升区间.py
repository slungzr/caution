import pywencai
import akshare as ak
import datetime
import pandas as pd
import os
import random
import time


# 读取最近生成的 '竞价爬升-*.csv' 文件，取出其中的最大日期（列名为 '日期'），
# 并将分析区间设为：从 today 到 csv_max_date（如果 csv_max_date < today，则交换顺序）
today = pd.Timestamp.today().normalize()

# 尝试查找最新的竞价爬升 CSV 文件并读取最大日期
csv_dir = os.path.abspath('.')
csv_pattern = '竞价爬升-20240504.csv'
csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.startswith('竞价爬升-20240504.csv') and f.endswith('.csv')]
csv_files = sorted(csv_files, key=os.path.getmtime, reverse=True)
csv_max_date = None
if csv_files:
    latest_csv = csv_files[0]
    # 尝试多种常见编码读取 CSV
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin1']
    tmp_df = None
    used_enc = None
    for enc in encodings:
        try:
            tmp_df = pd.read_csv(latest_csv, encoding=enc)
            used_enc = enc
            break
        except Exception:
            continue
    if tmp_df is not None:
        # print(f"read csv using encoding: {used_enc}")
        if '日期' in tmp_df.columns:
            # 优先按无分隔符的 YYYYMMDD 格式解析（例如 20240522），失败则回退到通用解析
            try:
                tmp_df['日期'] = pd.to_datetime(tmp_df['日期'].astype(str), format='%Y%m%d', errors='coerce')
                if tmp_df['日期'].isna().all():
                    tmp_df['日期'] = pd.to_datetime(tmp_df['日期'], errors='coerce')
            except Exception:
                tmp_df['日期'] = pd.to_datetime(tmp_df['日期'], errors='coerce')
            if tmp_df['日期'].notna().any():
                csv_max_date = tmp_df['日期'].max().normalize()
                # 要求：最大日期再减少一天
                csv_max_date = (csv_max_date - pd.Timedelta(days=1)).normalize()
        else:
            # 如果列名不匹配，可以在此处添加更多容错
            csv_max_date = None
    else:
        csv_max_date = None

# 如果没有找到或解析出错，回退到默认硬编码日期；否则使用 csv_max_date
if csv_max_date is None:
    # 回退（与原脚本接近的硬编码范围）
    # day 表示我们分析的结束日期（原来为 2025-09-16）
    day = datetime.date(2025, 9, 16)
    start_date = pd.Timestamp(datetime.date(2025, 9, 2))
else:
    # 将 today 与 csv_max_date 设为 start/end（确保 start <= end）
    if csv_max_date < today:
        start_date = csv_max_date
        end_date = today
    else:
        start_date = today
        end_date = csv_max_date

data = ak.tool_trade_date_hist_sina()
data['cal_date'] = data['trade_date']
data['trade_date'] = pd.to_datetime(data['trade_date'])
# 将 pretrade_date 列转换为 trade_date 列的上一行值
data['pretrade_date'] = data['trade_date'].shift(1)
data = data.sort_values('trade_date', ascending=False)

# 使用从 CSV 得到的 start/end 过滤数据（如果 csv_max_date 为 None 则使用原硬编码）
if csv_max_date is None:
    date_20220222 = datetime.date(2025, 9, 2)
    data = data[(data['trade_date'] <= pd.Timestamp(day)) & (data['trade_date'] >= pd.Timestamp(date_20220222))]
else:
    # 注意 data 的 trade_date 已经是 datetime 类型，在此按 start_date/end_date 过滤
    data = data[(data['trade_date'] >= pd.Timestamp(start_date)) & (data['trade_date'] <= pd.Timestamp(end_date))]
# 循环
for index, row in data.iterrows():
    if index == 0:
        continue
    # 获取当前行和上一行的 trade_date 值,转成m月d日格式，如8月2日
    tomorrow = row['trade_date'].strftime("%#m月%#d日")
    today = data.loc[index - 1, 'trade_date'].strftime("%#m月%#d日") 
    yesterday = data.loc[index - 2, 'trade_date'].strftime("%#m月%#d日")
    ereYesterday = data.loc[index - 3, 'trade_date'].strftime("%#m月%#d日")
    
    tomorrow_format = row['trade_date'].strftime("%Y%m%d")
    today_format = data.loc[index - 1, 'trade_date'].strftime("%Y%m%d")
    yesterday_format = data.loc[index - 2, 'trade_date'].strftime("%Y%m%d")
    ereYesterday_format = data.loc[index - 3, 'trade_date'].strftime("%Y%m%d")
    query='今日9点25分最低价>今日9点24分最高价，今日9点24分最低价≥今日9点23分最高价，今日9点23分最低价≥今日9点22分最高价，今日9点22分最低价≥今日9点21分最高价，今日9点21分最低价≥今日9点20分最高价,今日竞价涨幅,今日竞价换手率，今日最低价，昨日人气前100'
    query = query.replace('今日', today)
    query = query.replace('昨日', yesterday)
    query = query.replace('前日', ereYesterday)
    query1='昨日人气前100，明日和今日收盘价，明日和今日开盘价，昨日和前日成交量，今日和昨日涨跌幅，今日上市天数大于3，昨日非跌停，昨日连扳天数,昨日实体涨跌幅，前日实体涨跌幅，昨日换手率,前日人气排名,昨日ddx，今日实体涨跌幅，今日竞价金额>一千万'
    query1 = query1.replace('昨日', yesterday)
    query1 = query1.replace('明日', tomorrow)
    query1 = query1.replace('今日', today)
    query1 = query1.replace('前日', ereYesterday)
    
    print(query)
    print(query1)

    try:
        cookie='other_uid=Ths_iwencai_Xuangu_sm79wfgix7t3519vltpjzuq9bdk8li6a; u_ukey=A10702B8689642C6BE607730E11E6E4A; u_uver=1.0.0; u_dpass=z6QA5xYLOpYlD%2Ff93a3QmBUSKed47Fn5t6O740k%2BrU2QFqx%2BcNnm2HkWtm4geczTHi80LrSsTFH9a%2B6rtRvqGg%3D%3D; u_did=7B661B723734467F89AFC29A49CB9F16; u_ttype=WEB; user=MDpteF8xNzgyNTA0NDM6Ok5vbmU6NTAwOjE4ODI1MDQ0Mzo1LDEsNDA7NiwxLDQwOzcsMTExMTExMTExMTEwLDQwOzgsMTExMTAxMTEwMDAwMTExMTEwMDEwMDEwMDEwMDAwMDAsNDA7MzMsMDAwMTAwMDAwMDAwLDkxOzM2LDEwMDExMTExMDAwMDExMDAxMDExMTExMSw5MTs0NiwwMDAwMTExMTEwMDAwMDExMTExMTExMTEsOTE7NTEsMTEwMDAwMDAwMDAwMDAwMCw5MTs1OCwwMDAwMDAwMDAwMDAwMDAwMSw5MTs3OCwxLDkxOzg3LDAwMDAwMDAwMDAwMDAwMDAwMDAxMDAwMCw5MTsxMTksMDAwMDAwMDAwMDAwMDAwMDAwMTAxMDAwMDAwMDAwMDAwMDAwMDAwMDAsOTE7MTI1LDExLDkxOzEzMCwxMDEwMDAwMDAwMDAwLDkxOzQ0LDExLDQwOzEsMTAxLDQwOzIsMSw0MDszLDEsNDA7MTAyLDEsNDA6MTY6OjoxNzgyNTA0NDM6MTc3NjQzODA1OTo6OjEzODU3Nzk4NjA6NjA0ODAwOjA6MTE0ODkzZDQ5YmE1ZjIwZTBjOGZmNzg1YmM3OGQ0MGQ0OmRlZmF1bHRfNTox; userid=178250443; u_name=mx_178250443; escapename=mx_178250443; ticket=1a7af4be917212772308dd4d7e9b533c; user_status=0; utk=4516ecbe8ba8eb7138c87ce8482fd260; sess_tk=eyJ0eXAiOiJKV1QiLCJhbGciOiJFUzI1NiIsImtpZCI6InNlc3NfdGtfMSIsImJ0eSI6InNlc3NfdGsifQ.eyJqdGkiOiJkNDQwOGRjNzViNzhmZmM4ZTAyMDVmYmE0OTNkODkxNDEiLCJpYXQiOjE3NzY0MzgwNTksImV4cCI6MTc3NzA0Mjg1OSwic3ViIjoiMTc4MjUwNDQzIiwiaXNzIjoidXBhc3MuaXdlbmNhaS5jb20iLCJhdWQiOiIyMDIwMTExODUyODg5MDcyIiwiYWN0Ijoib2ZjIiwiY3VocyI6IjcyYWUwNTA2MWJmNDhmMzk0MGFlMmMxYTJiMTM0MTU2OGQ3ZTc2NzJlOWFhNTMyMzY3NTlkYTRmNDJmNzQ4ZTQifQ.0GGKkUk9AmPItIXbQN1t1ex05kLbTZl5G8YNqPLeKemxYx5aBtqdgfi_Ea0u0eRE_JgPoW344ya3kC7AVMkVJQ; cuc=p1tvz7muclod; _clck=ylwejl%7C2%7Cg5c%7C0%7C0; THSSESSID=c615304b67aed5e410a7fade0d; _clsk=324wpk3i7lj7%7C1776597908837%7C5%7C1%7C; v=A03870DNGMjDB7zkRBXIprpcXGLCKoFMC1zl0I_SizRSPWOcV3qRzJuu9Ycc'
       
        # Add random delay between requests to avoid rate limiting
        time.sleep(random.uniform(1, 10))
        result = pywencai.get(query=query, cookie=cookie)
        result1 = pywencai.get(query=query1, cookie=cookie)


        if result is None or isinstance(result, dict) or result1 is None or isinstance(result1, dict) or (hasattr(result, 'empty') and result.empty) or (hasattr(result1, 'empty') and result1.empty):
            continue
            
        # 获取两个DataFrame的交集，同名列只保留一个
        common_stocks = pd.merge(result, result1, on='股票代码', how='inner', suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
      
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
       
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')

        if common_stocks.empty:
            continue
        # 将包含竞价强度的列重命名为'竞价强度'
    
        # 去掉列名中的日期格式 [yyyymmdd]，替换为'今日'
        common_stocks.columns = common_stocks.columns.str.replace(r'\[' + today_format + r'\]', '今日', regex=True)
        # 去掉列名中的日期格式 [yyyymmdd]，替换为'昨日'和'明日'
        common_stocks.columns = common_stocks.columns.str.replace(r'\[' + yesterday_format + r'\]', '昨日', regex=True)
        common_stocks.columns = common_stocks.columns.str.replace(r'\[' + tomorrow_format + r'\]', '明日', regex=True)
        common_stocks.columns = common_stocks.columns.str.replace(r'\[' + ereYesterday_format + r'\]', '前日', regex=True)
        # Get the column name that contains '竞价强度'
    
        # 计算收益率
        common_stocks['收益率'] = (common_stocks['收盘价:不复权明日'].astype(float) - common_stocks['开盘价:不复权今日'].astype(float)) / common_stocks['开盘价:不复权今日'].astype(float)
        common_stocks['开盘收益率'] =  (common_stocks['开盘价:不复权明日'].astype(float) - common_stocks['开盘价:不复权今日'].astype(float)) / common_stocks['开盘价:不复权今日'].astype(float)
        
        # 计算量比
        common_stocks['量比'] = common_stocks['成交量昨日'].astype(float) / common_stocks['成交量前日'].astype(float)
        
        # 添加日期列
        common_stocks['日期'] = today_format
    # 找出包含'实体涨跌幅'和'昨日'的列，重命名为'实体涨跌幅昨日'
        common_stocks.columns = common_stocks.columns.str.replace(r'实体涨跌幅.*昨日', '实体涨跌幅昨日', regex=True)
        # Find column with '阳线' and rename it to match the pattern
        common_stocks.columns = common_stocks.columns.str.replace(r'技术形态.*', '技术形态昨日', regex=True)

        # 只保留指定的列
        # Calculate the percentage drop from open price
        common_stocks['开盘后最大跌幅'] = (common_stocks['最低价:不复权今日'].astype(float) - common_stocks['开盘价:不复权今日'].astype(float)) / common_stocks['开盘价:不复权今日'].astype(float)
        
        columns_to_keep = ['日期', '股票代码', '股票简称', '开盘价:不复权今日', '收盘价:不复权今日','开盘价:不复权明日', '收盘价:不复权明日', 
                        '涨跌幅:前复权今日', '涨跌幅:前复权昨日','开盘收益率','收益率', '量比', '个股热度排名昨日','连续涨停天数昨日',
                        '竞价涨幅今日','实体涨跌幅今日','实体涨跌幅昨日','实体涨跌幅前日','成交量昨日', '成交量前日','换手率昨日','个股热度昨日','个股热度前日',
                        '大单动向(ddx值)昨日','最低价:不复权今日', '开盘后最大跌幅']
        common_stocks = common_stocks[columns_to_keep]
        common_stocks['个股热度排名昨日'] = common_stocks['个股热度排名昨日'].apply(lambda x: int(x.split('/')[0]) if isinstance(x, str) and '/' in x else x)
        common_stocks = common_stocks[common_stocks['个股热度排名昨日'] <= 100]
        
        # 将结果追加到CSV文件
        # 最终写回的文件名：优先使用从现有竞价爬升 CSV 读取到的最大日期（csv_max_date），
        # 如果没有找到则回退到原来的硬编码日期 20250902
        if 'csv_max_date' in globals() and csv_max_date is not None:
            out_date_str = pd.to_datetime(csv_max_date).strftime('%Y%m%d')
        else:
            out_date_str = datetime.date(2025, 9, 2).strftime('%Y%m%d')
        csv_file = f'竞价爬升-{out_date_str}.csv'
        # 如果文件不存在，创建文件并写入表头
        if not os.path.exists(csv_file):
            # 使用 gbk 编码以在中文 Windows/Excel 中避免乱码
            common_stocks.to_csv(csv_file, index=False, mode='w', encoding='gbk')
        else:
            # 如果文件存在，追加数据不写入表头
            common_stocks.to_csv(csv_file, index=False, mode='a', header=False, encoding='gbk')

        # 另外，按要求也把结果追加到固定文件 '竞价爬升-20240504.csv'
        fixed_csv = '竞价爬升-20240504.csv'
        if not os.path.exists(fixed_csv):
            common_stocks.to_csv(fixed_csv, index=False, mode='w', encoding='gbk')
        else:
            common_stocks.to_csv(fixed_csv, index=False, mode='a', header=False, encoding='gbk')
        print(common_stocks.columns)
    except Exception as e:
            print(f"Error fetching data: {e}")
            continue
