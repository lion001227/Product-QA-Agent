from langchain.tools import tool
import requests
from datetime import datetime
import random
import json

@tool
def profile_search(question:str)->str:
    """
    查询交易所文档，回答产品相关问题
    当用户咨询交易所文档相关内容时使用
    """
    from rag.rag import rag_qa
    result=rag_qa(question)
    return result


@tool
def return_document_sources(question:str)->str:
    """查询回答的文档依据、文件名、页码和相关原文
        当用户询问“来源是什么”“依据在哪”“原文在哪”“第几页”
        或希望核对回答时使用
    """
    from rag.rag import search_document_sources
    result=search_document_sources(question)
    return result



#上交所地址
SSE_LATEST_ANNOUNCEMENTS_URL = (
    "https://www.sse.com.cn/disclosure/listedinfo/announcement/"
    "json/stock_bulletin_publish_order.json"
)



#深交所地址
SZSE_GENERAL_ANNOUNCEMENTS_URL=("https://www.szse.cn/api/search/content")



def _normalize_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    if raw_url.startswith("//"):
        return "https:" + raw_url
    if raw_url.startswith("/"):
        return "https://www.sse.com.cn" + raw_url
    if raw_url.startswith("http://"):
        return "https://" + raw_url[len("http://"):]
    return raw_url


@tool
def sse_latest_announcements(
    stock_code: str = "",
    keyword: str = "",
    limit: int = 5,
) -> str:
    """
    实时查询上海证券交易所官网的上市公司最新公告。适合于用户询问：
    - 上交所最新公告
    - 某只上交所股票最新公告
    - 最近有什么公告
    - 与某股票或关键词相关的最新上交所公告

    注意：接口本身只返回最新公告列表（不支持服务端按代码/关键词过滤），
    本函数会拉取列表后在本地做筛选。因此若某股票近期没有公告，
    或列表条数不够多导致没覆盖到，也可能筛不出结果。

    参数：
    stock_code: 可选的6位股票代码，例如"600519""601318""688001"
    keyword: 可选关键词，例如"回购""分红""股权激励"
    limit：最多返回的公告数，默认是5，最多是10
    """
    limit = max(1, min(limit, 10))
    stock_code = stock_code.strip()
    keyword = keyword.strip()
    _HEADERS = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0"
        ),
    }

    if stock_code and (len(stock_code) != 6 or not stock_code.isdigit()):
        return "上交所股票代码格式错误，应为6位数字，例如：600519"

    params = {"v": random.random()}

    try:
        resp = requests.get(
            SSE_LATEST_ANNOUNCEMENTS_URL,
            headers=_HEADERS,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"无法访问上交所公告接口：{e}"
    except (ValueError, json.JSONDecodeError):
        return "上交所返回的数据不是有效的JSON"

    announcements = data.get("publishData", [])
    if not isinstance(announcements, list) or not announcements:
        return "上交所公告接口暂无数据返回"

    # 本地筛选
    filtered = []
    for item in announcements:
        if stock_code and item.get("securityCode") != stock_code:
            continue
        if keyword and keyword not in item.get("bulletinTitle", ""):
            continue
        filtered.append(item)

    if not filtered:
        conditions = []
        if stock_code:
            conditions.append(f"股票代码：{stock_code}")
        if keyword:
            conditions.append(f"关键词：{keyword}")
        suffix = f"（{'，'.join(conditions)}）" if conditions else ""
        return (
            f"在最新公告列表中未找到匹配结果{suffix}。"
            f"该接口只返回最新一批公告（共{len(announcements)}条），"
            f"如目标股票/关键词近期没有公告，可能无法查到。"
        )

    results = []
    for item in filtered[:limit]:
        title = item.get("bulletinTitle", "").strip()
        if not title:
            continue
        code = item.get("securityCode", "")
        name = item.get("securityAbbr", "")
        publish_date = item.get("discloseDate", "日期未识别")
        bulletin_type = item.get("bulletinClassic", "")
        url = _normalize_url(item.get("bulletinUrl", ""))

        block = (
            f"【公告{len(results) + 1}】\n"
            f"股票代码：{code}\n"
            f"股票简称：{name}\n"
            f"标题：{title}\n"
            f"发布日期：{publish_date}\n"
        )
        if bulletin_type:
            block += f"公告类型：{bulletin_type}\n"
        block += f"官网链接：{url}"

        results.append(block)

    return "\n\n".join(results) if results else "未找到匹配的上交所最新公告"



@tool
def szse_latest_announcements(keyword:str="",limit:int=5,)->str:
    """
        实时查询深交所官网的一般公告。适合于用户询问：
        - 深交所最新公告
        - 最近有什么公告
        - 与某关键词相关的最新深交所公告

        参数：
        keyword:可选关键词，例如“报价回购”“基金”“债券”
        limit：最多返回的公告数，默认是5，最多是10
    """
    limit=max(1,min(limit,10))

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://www.szse.cn",
        "Referer": (
            "https://www.szse.cn/disclosure/notice/general/index.html"
        ),
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0"
        ),
        "X-Request-Type": "ajax",
        "X-Requested-With": "XMLHttpRequest",
    }

    # 深交所接口要求的 POST 参数
    data = {
        "keyword": keyword,
        "time": 0,
        "range": "title",
        "channelCode[]": "general_news",
        "currentPage": 1,
        "pageSize": limit,
        "scope": 0,
        "endTime": 0,
    }

    try:
        response=requests.post(
            SZSE_GENERAL_ANNOUNCEMENTS_URL,
            params={
                "random":__import__("random").random()
            },
            headers=headers,
            data=data,
            timeout=60,)

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.RequestException as e:
        return f"无法访问深交所公告接口：{e}"

    except ValueError as e:
        return f"深交所返回的数据不是有效的JSON"

    #获取公告列表
    announcements=result.get("data",[])
    if not announcements:
        suffixes=f"关键字{keyword}"if keyword else ""
        return f"未找到匹配的深交所公告{suffixes}"

    results=[]

    for item in announcements:
        title=item.get("doctitle","").strip()

        if not title:
            continue

        #公告日期
        publish_time=item.get("docpubtime")
        if publish_time:
            try:
                publish_date=datetime.fromtimestamp(publish_time/1000).strftime("%Y-%m-%d")
            except (ValueError,TypeError,OSError) :
                publish_date="日期未识别"
        else:
            publish_date = "日期未识别"

        #官网公告链接
        url=item.get("docpuburl","")


        #有些接口可能返回http，统一转换成https
        if url.startswith("http://"):
            url="https://"+url[len("http://"):]


        results.append(
            f"【公告{len(results)+1}】\n"
            f"标题：{title}\n"
            f"发布日期{publish_date}\n"
            f"官网链接{url}"
        )

        #达到数量限制
        if len(results) >= limit:
            break

    if not results:
        suffix=f"(关键词：{keyword})" if keyword else ""
        return f"未找到匹配的深交所最新公告{suffix}"


    return "\n\n".join(results)



# result = szse_latest_announcements.invoke({
#     "keyword": "",
#     "limit": 5
# })
#
# print(result)

# if __name__ == "__main__":
#     print(
#         sse_latest_announcements.invoke(
#             {
#                 "stock_code": "",
#                 "keyword": "",
#                 "limit": 5,
#             }
#         )
#     )