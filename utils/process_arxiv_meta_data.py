import sys
import json
import os


def single_process_arxiv_metadata(item, corpus_id):
    # 提取paper_id
    paper_id = item["id"]

    # 提取标题
    title = item["title"]

    # 提取摘要
    abstract = f"<|reference_start|>{item['abstract']}<|reference_end|>"

    # 构建作者字符串
    if "authors_parsed" in item and item["authors_parsed"]:
        authors = []
        for author in item["authors_parsed"]:
            if author[0] and author[1]:  # 如果有姓和名
                authors.append(f"{author[1]} {author[0]}")
        authors_str = " and ".join(authors)
    else:
        authors_str = item["authors"]

    # 确定年份
    year = ""
    if "versions" in item and item["versions"]:
        version_date = item["versions"][0]["created"]
        # 格式如: "Mon, 30 Apr 2007 20:32:04 GMT"
        import re
        year_match = re.search(r'\d{4}', version_date)
        if year_match:
            year = year_match.group(0)

    # 生成引用键
    # 使用第一个作者的姓氏（如有）
    first_author_surname = ""
    if "authors_parsed" in item and item["authors_parsed"] and item["authors_parsed"][0]:
        first_author_surname = item["authors_parsed"][0][0].lower()
    else:
        # 尝试从原始作者字符串中提取
        author_parts = item["authors"].split()[0].lower()
        first_author_surname = author_parts

    citation_key = f"{first_author_surname}{year}{title.split()[0].lower()}"

    # 构建BibTeX
    categories = item.get("categories", "").replace(" ", ".")
    primary_class = categories.split()[0] if categories else ""
    secondary_classes = " ".join(categories.split()[1:]) if len(categories.split()) > 1 else ""

    bibtex = f"@article{{{citation_key},\n"
    bibtex += f"    title={{{title}}},\n"
    bibtex += f"    author={{{authors_str}}},\n"
    bibtex += f"    journal={{arXiv preprint arXiv:{paper_id}}},\n"
    bibtex += f"    year={{{year}}},\n"
    bibtex += f"    archivePrefix={{arXiv}},\n"
    bibtex += f"    eprint={{{paper_id}}},\n"

    if primary_class:
        if secondary_classes:
            bibtex += f"    primaryClass={{{primary_class} {secondary_classes}}}\n"
        else:
            bibtex += f"    primaryClass={{{primary_class}}}\n"

    bibtex += "}"

    # 构建输出
    result = {
        "corpus_id": corpus_id,
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "source": "arxiv",
        "bibtex": bibtex,
        "citation_key": citation_key
    }

    return result


def process_meta_data(input_meta_file, output_corpus_file):
    count = 0
    output_corpus_data = []
    with open(input_meta_file, "r") as fi:
        for line in fi:
            curr_meta_data = json.loads(line)
            corpus_id = f"arxiv-{str(count)}"
            curr_corpus_item = single_process_arxiv_metadata(curr_meta_data, corpus_id)
            output_corpus_data.append(curr_corpus_item)
            count += 1
            if count > 1000:
                break
    with open(output_corpus_file, "w") as fo:
        for each in output_corpus_data:
            fo.write(json.dumps(each) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python script.py <input file path> [output file path]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_meta_data(input_file, output_file)




