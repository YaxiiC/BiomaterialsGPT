### 📘 PubMed 批量查询与全文抓取工具
##### 功能：限定期刊 + 关键词搜索，获取元数据+摘要+全文链接+PMC全文

from Bio import Entrez, Medline
import pandas as pd
import time
import os
from tqdm import tqdm
import requests


# 配置：邮箱和 API Key
Entrez.email = "guojy23@m.fudan.edu.cn"  # 🔁 请替换为你自己的邮箱
Entrez.api_key = "7eab3c9283e032b0a5b3451997ae12fb2208"

# 创建保存目录
os.makedirs("results", exist_ok=True)
os.makedirs("results/pmc_xml", exist_ok=True)

# ===== 1. 期刊列表（原始） =====
journal_list_raw = [
    "Bioactive Materials", "Biomaterials", "Advanced Healthcare Materials",
    "Acta Biomaterialia", "Materials Today Bio", "Biofabrication",
    "Journal of Materials Chemistry B", "Biomaterials Science", "Regenerative Biomaterials",
    "ACS Biomaterials Science & Engineering", "Colloids and Surfaces B-Biointerfaces",
    "Tissue Engineering Part B-Reviews", "Journal of Functional Biomaterials",
    "Journal of Bionic Engineering", "Dental Materials",
    "Artificial Cells Nanomedicine and Biotechnology", "Macromolecular Bioscience",
    "Journal of Materials Science-Materials in Medicine", "Biomedical Materials",
    "Journal of Biomedical Materials Research Part A", "Journal of Biomaterials Science-Polymer Edition",
    "Tissue Engineering Part A", "Biomimetics",
    "Journal of the Mechanical Behavior of Biomedical Materials",
    "Journal of Biomedical Materials Research Part B-Applied Biomaterials",
    "Journal of Applied Biomaterials & Functional Materials", "Tissue Engineering Part C-Methods",
    "International Journal of Polymeric Materials and Polymeric Biomaterials",
    "Journal of Biomaterials Applications", "Journal of Bioactive and Compatible Polymers",
    "Dental Materials Journal", "Journal of Oral Science", "Biointerphases",
    "Bioinspired Biomimetic and Nanobiomaterials", "Bio-Medical Materials and Engineering"
]

# ===== 2. 查询关键词与时间范围 =====
search_query = """
(biocompatible[Title/Abstract] OR cytocompatible[Title/Abstract])
AND (hydrogel[Title/Abstract] OR "gel polymer"[Title/Abstract])
AND ("drug delivery"[Title/Abstract] OR therapeutic[Title/Abstract])
AND ("2014"[Date - Publication] : "2024"[Date - Publication])
"""

# ===== 3. 批量查询函数 =====
def search_pubmed(journal_name, retmax=1000):
    full_query = f"{search_query} AND \"{journal_name}\"[Journal]"
    handle = Entrez.esearch(db="pubmed", term=full_query, retmax=retmax)
    record = Entrez.read(handle)
    return record["IdList"]

# ===== 4. 获取文献详情 =====
def fetch_details(id_list):
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    return records

# ===== 5. 获取 PMC 全文 XML（如有） =====
def fetch_pmc_xml(pmid):
    try:
        link_handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
        links = Entrez.read(link_handle)
        pmc_id = links[0]['LinkSetDb'][0]['Link'][0]['Id']
        xml_handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
        xml_data = xml_handle.read()
        with open(f"results/pmc_xml/{pmid}.xml", "w", encoding="utf-8") as f:
            f.write(xml_data)
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    except:
        return ""


# ===== 6. 主流程 =====
for journal in tqdm(journal_list_raw):
    print(f"\n🔍 正在查询期刊：{journal}")
    pmids = search_pubmed(journal, retmax=1000)
    time.sleep(1)
    if not pmids:
        continue

    all_data = []
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i+200]
        details = fetch_details(batch)
        for rec in details:
            pmid = rec.get("PMID", "")
            title = rec.get("TI", "")
            abstract = rec.get("AB", "")
            journal_name = rec.get("JT", "")
            pubdate = rec.get("DP", "")
            types = "; ".join(rec.get("PT", []))
            doi = rec.get("LID", "").split()[0] if "LID" in rec else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            pmc_link = fetch_pmc_xml(pmid)

            all_data.append({
                "PMID": pmid,
                "Title": title,
                "Abstract": abstract,
                "Journal": journal_name,
                "Date": pubdate,
                "Types": types,
                "DOI": doi,
                "PubMed_Link": link,
                "PMC_Link": pmc_link
            })
        time.sleep(1)

    # 保存为 CSV
    df = pd.DataFrame(all_data)
    save_path = f"results/{journal.replace(' ', '_')}.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ 已保存：{save_path}")
