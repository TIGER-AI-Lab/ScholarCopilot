from datetime import datetime
import tempfile
from scholar_copilot_model import *
import torch
import faiss
import time
import gradio as gr


def generate_citation(input_text):
    global index
    new_input_text = input_text + " <|cite_start|>"
    new_input = tokenizer(new_input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        new_output = model(
            new_input.input_ids,
            attention_mask=new_input.attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
    cite_rep = new_output.hidden_states[-1][:, -1, :]
    retrieved_k_results = retrieve_reference(index, lookup_indices, cite_rep, top_k=10)
    searched_citations = []
    for each in retrieved_k_results:
        curr_index, distance = each
        print("index", curr_index)
        if curr_index not in meta_data:
            print("index not found in meta_data", curr_index)
            continue
        paper_id = meta_data[curr_index]["paper_id"]
        print("paper_id", paper_id)
        citation_info = citation_map_data[paper_id]
        print("generate_citation citation_info", citation_info)
        searched_citations.append(citation_info)
    return searched_citations


def split_yield_list(input_text, prefix_length):
    prefix_text = input_text[:prefix_length]
    text = input_text[prefix_length:]
    text_list = text.split(" ")
    return prefix_text, text_list


def check_3_sentence(display_text):
    if display_text.endswith('.'):
        return display_text
    end_index = display_text.rfind('.\n')
    return display_text[: end_index + 1]


def stream_complete_3_sentence(text, citations_data, progress=gr.Progress()):
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
    for each in yield_list:
        if "." in each and (each.endswith(".") or ".\n" in each):
            sentence_num += 1
            print("sentence_num: ", sentence_num, "each", each)
        curr_yield_text += " " + each
        yield curr_yield_text, citations_data
        if sentence_num == 3:
            enough = True
            display_text = curr_yield_text
            display_text = check_3_sentence(display_text)
            break
        time.sleep(0.1)
    curr_prefix_length = len(curr_yield_text)
    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        citations_data += citation_data_list
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
        for each in yield_list:
            if "." in each and (each.endswith(".") or ".\n" in each):
                sentence_num += 1
                print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
            yield curr_yield_text, citations_data
            if sentence_num == 3:
                enough = True
                display_text = curr_yield_text
                display_text = check_3_sentence(display_text)
                break
            time.sleep(0.1)
        curr_prefix_length = len(curr_yield_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    yield display_text, citations_data
    time.sleep(0.1)


def stream_generate(text, citations_data, progress=gr.Progress()):
    sentence_num = 0
    enough = False
    current_text = text
    current_text = preprocess_input_text(current_text)
    display_text = current_text.replace("<|paper_start|> ", "")
    curr_prefix_length = len(display_text)
    current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
    reference_id_list = []
    display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
    # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
    for each in yield_list:
        if "." in each and (each.endswith(".") or ".\n" in each):
            sentence_num += 1
            print("sentence_num: ", sentence_num, "each", each)
        curr_yield_text += " " + each
        yield curr_yield_text, citations_data
        time.sleep(0.1)
    curr_prefix_length = len(curr_yield_text)
    while cite_start_hidden_state is not None and not enough:
        retrieved_k_results = retrieve_reference(index, lookup_indices, cite_start_hidden_state, top_k=1)
        reference, curr_index = llm_rerank(retrieved_k_results, meta_data)
        reference_id_list.append(curr_index)
        current_text = current_text + reference
        current_text, cite_start_hidden_state = single_complete_step(model, tokenizer, device, current_text)
        display_text, citation_data_list = replace_citations(current_text, reference_id_list, citation_map_data)
        citations_data += citation_data_list
        curr_yield_text, yield_list = split_yield_list(display_text, curr_prefix_length)
        # print("curr_yield_text, yield_list", curr_yield_text, yield_list)
        for each in yield_list:
            if "." in each and (each.endswith(".") or ".\n" in each):
                sentence_num += 1
                print("sentence_num: ", sentence_num, "each", each)
            curr_yield_text += " " + each
            yield curr_yield_text, citations_data
            time.sleep(0.1)
        curr_prefix_length = len(curr_yield_text)
    display_text, citation_data_list = post_process_output_text(display_text, reference_id_list, citation_map_data)
    citations_data += citation_data_list
    yield display_text, citations_data
    time.sleep(0.1)


def format_citation(citation_key, url):
    total_length = 150
    citation_length = len(citation_key)
    url_length = len(url)
    if citation_length > 110:
        citation_key = citation_key[:105] + "...  "
        citation_length = 110
    return citation_key + " " * (total_length - citation_length - url_length) + url


def search_and_show_citations(input_text):
    curr_citations_data = generate_citation(input_text)
    curr_search_candidates = curr_citations_data
    choices = []
    for cit in curr_citations_data:
        paper_id = cit["id"]
        citation_key = cit["citation_key"]
        title = cit["title"].replace("\n", " ").replace("  ", " ")
        url = f" (https://arxiv.org/abs/{paper_id})"
        item = format_citation(citation_key + ": " + title, url)
        # print("item", item)
        choices.append(item)
    # return {
    #     citation_box: gr.Group(visible=True),
    #     citation_checkboxes: gr.CheckboxGroup(
    #         choices=choices,
    #         value=[],
    #     ),
    #     curr_search_candidates: curr_search_candidates
    # }
    return gr.Group(visible=True), gr.CheckboxGroup(choices=choices, value=[]), curr_search_candidates


def insert_selected_citations(text, selected_citations, citations_data, curr_search_candidates):
    if not selected_citations:
        return text

    selected_citations = [each.split(": ")[0] for each in selected_citations]
    citations = ", ".join(selected_citations)
    new_text = text + " \\cite{" + citations + "}"
    for each_candidate in curr_search_candidates:
        if each_candidate["citation_key"] in selected_citations:
            citations_data.append(each_candidate)
    return new_text


def update_bibtex(citations_data):
    # print("citations_data", citations_data)
    if not citations_data:
        return None  # 如果没有引用历史，返回None

    bibtex_entries = []
    for cit in citations_data:
        if cit["bibtex"] not in bibtex_entries:
            bibtex_entries.append(cit["bibtex"])
    content = "\n\n".join(bibtex_entries)
    return content


def clear_cache(citations_data, curr_search_candidates):
    # citations_data = []
    # curr_search_candidates = []
    citations_checkbox = gr.CheckboxGroup(
        choices=[],
        value=[],
    )
    return "", citations_checkbox, "", [], []


def load_example(file_name=""):
    example_text = ""
    with open(f"src/{file_name}", "r") as fi:
        for line in fi.readlines():
            example_text += line
    return example_text


def load_example_text(choice):
    if choice == "Template":
        return load_example("template.txt")
    elif choice == "Example 1":
        return load_example("mmlu-pro-example.txt")
    elif choice == "Example 2":
        return load_example("harness-example.txt")
    elif choice == "Example 3":
        return load_example("vlm2vec-example.txt")


with gr.Blocks(css="""
    :root {
        --color-1: #89A8B2;
        --color-2: #F1F0E8; 
        --color-3: #B3C8CF;
        --color-4: #E5E1DA;
    }
    .container {
        max-width: 1200px;
        margin: auto;
        padding: 20px;
        background-color: var(--color-2);
    }
    .header {
        text-align: center;
        margin-bottom: 40px;
        background: linear-gradient(135deg, var(--color-1), var(--color-3));
        padding: 30px;
        border-radius: 15px;
        color: var(--color-4);
    }
    .intro-section {
        background: var(--color-4);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-list {
        background: var(--color-4);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .main-editor {
        background: var(--color-4);
        padding: 0px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .button-row {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    .button-row button, .button-row a {
        flex: 1;
        min-width: 200px;
        background: var(--color-1);
        border: none;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        text-decoration: none;
        text-align: center;
    }
    .button-row button:hover, .button-row a:hover {
        background: var(--color-4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .citation-section {
        background: var(--color-4);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 1000px; 
        margin-left: auto; 
        margin-right: auto;
    }
    .citation-section button {
        background: var(--color-1);
        border: none;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .citation-section button:hover {
        background: var(--color-4);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .citation-section .gr-form {
        max-width: 100%;
    }
    .citation-section .gr-checkbox-group {
        max-width: 100%;
    }
    .textbox textarea {
        border: 2px solid var(--color-2);
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .textbox textarea:focus {
        border-color: var(--color-1);
        outline: none;
    }
    .checkbox-group {
        background: var(--color-3);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .bibtex-section {
        background: var(--color-4);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .bibtex-display {
        font-family: monospace;
        white-space: pre-wrap;
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--color-1);
        margin-top: 10px;
    }
    .example-selector {
        margin-bottom: 20px;
    }
    .example-selector select {
        width: 100%;
        padding: 10px;
        border: 2px solid var(--color-1);
        border-radius: 8px;
        background-color: white;
        font-size: 16px;
        color: #333;
        cursor: pointer;
    }
    .example-selector select:hover {
        border-color: var(--color-3);
    }
    .example-selector select:focus {
        outline: none;
        border-color: var(--color-1);
        box-shadow: 0 0 5px rgba(137, 168, 178, 0.3);
    }
""") as app:
    citations_data = gr.State([])
    curr_search_candidates = gr.State([])
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("""
                <style>
                .title-row {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 20px;
                    padding: 0 20px;
                }
                .title {
                    flex: 0.6;
                    text-align: center;
                }
                .subtitle {
                font-size: 1.2em;
                color: #666;
                text-align: center;
                margin-top: 5px;
                font-weight: normal;
                }
                </style>
            """)
            with gr.Row(elem_classes="title-row", equal_height=True):
                gr.Markdown(
                    """<h1 style='font-size: 2.5em; margin: 0; padding: 0;'>Scholar Copilot</h1>""",
                    elem_classes="title"
                )
            gr.Markdown(
                """<h3 class='subtitle'> Your Academic Writing Assistant -- By <a href="https://huggingface.co/TIGER-Lab" target="_blank">TIGER-Lab</a></h3>
                   <p>Authors: Yubo Wang, Xueguang Ma, Xiang Yue, and Wenhu Chen</p>
                   <p>Contact: <a href="mailto:yubo.wang.sunny@gmail.com">yubo.wang.sunny@gmail.com</a></p>
                   <p>To set up the ScholarCopilot demo on your own server, visit <a href="https://github.com/TIGER-AI-Lab/ScholarCopilot" target="_blank">https://github.com/TIGER-AI-Lab/ScholarCopilot</a></p>"""
            )


        # Introduction section
        with gr.Column(elem_classes="intro-section"):
            gr.Markdown("""
                Scholar Copilot improves the academic writing process by seamlessly integrating automatic text completion and intelligent citation suggestions into a cohesive, human-in-the-loop AI-driven pipeline. Designed to enhance productivity and creativity, it provides researchers with high-quality text generation and precise citation recommendations powered by iterative and context-aware Retrieval-Augmented Generation (RAG).

                The current version of Scholar Copilot leverages a state-of-the-art 7-billion-parameter language model (LLM) trained on the complete Arxiv full paper corpus. This unified model for retrieval and generation is adept at making context-sensitive decisions about when to cite, what to cite, and how to generate coherent content based on reference papers.
            """)

            with gr.Column(elem_classes="feature-list"):
                gr.Markdown("""
                    ### 🚀 Core Features:

                    * 📝 **Next-3-Sentence Suggestions**: Facilitates writing by predicting the next sentences with automatic retrieval and citation of relevant reference papers.
                    * 📚 **Citation Suggestions on Demand**: Provides precise, contextually appropriate paper citations whenever needed.
                    * ✨ **Full Section Auto-Completion**: Assists in brainstorming and drafting comprehensive paper content and structure.
                """)

            gr.Markdown("""The current version of ScholarCopilot primarily focuses on the introduction and related work sections of computer science academic papers. We will support full-paper writing in future releases.""")

        example_text = load_example("template.txt")
        # Main editor section
        with gr.Column(elem_classes="main-editor"):
            example_selector = gr.Dropdown(
                choices=["Template", "Example 1", "Example 2", "Example 3"],
                value="Default",
                label="Choose an example:",
                elem_classes="example-selector"
            )
            text_input = gr.Textbox(
                lines=20,
                label="Write your paper here",
                placeholder="Start writing your academic paper...",
                elem_classes="textbox",
                value=example_text
            )
            # file_output = gr.File(visible=False)
            with gr.Row(elem_classes="button-row"):
                complete_btn = gr.Button("🔄 Complete 3 sentences", size="md")
                generate_btn = gr.Button("✨ Generate to the end", size="md")
                citation_btn = gr.Button("📚 Search citations", size="md")
                update_bibtex_btn = gr.Button("📝 Update BibTeX", size="md")
                clear_btn = gr.Button("🗑️ Clear All", size="md")

        # Citation section
        with gr.Column(elem_classes="citation-section"):
            citation_box = gr.Group(visible=True)
            with citation_box:
                gr.Markdown("### 📚 Citation Suggestions")
                citation_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="Select citations to insert",
                    interactive=True
                )
                insert_citation_btn = gr.Button("📎 Insert selected citations", size="lg")

                gr.Markdown("### 📝 Existing BibTeX Entries")
                bibtex_display = gr.TextArea(
                    label="BibTeX",
                    interactive=False,
                    elem_classes="bibtex-display"
                )

        # Event handlers
        complete_btn.click(
            fn=stream_complete_3_sentence,
            inputs=[text_input, citations_data],
            outputs=[text_input, citations_data],
            queue=True
        )

        generate_btn.click(
            fn=stream_generate,
            inputs=[text_input, citations_data],
            outputs=[text_input, citations_data],
            queue=True
        )

        citation_btn.click(
            fn=search_and_show_citations,
            inputs=[text_input],
            outputs=[citation_box, citation_checkboxes, curr_search_candidates]
        )

        insert_citation_btn.click(
            fn=insert_selected_citations,
            inputs=[text_input, citation_checkboxes, citations_data, curr_search_candidates],
            outputs=[text_input]
        )

        clear_btn.click(
            fn=clear_cache,
            inputs=[citations_data, curr_search_candidates],
            outputs=[text_input, citation_checkboxes, bibtex_display, citations_data, curr_search_candidates]
        )
        update_bibtex_btn.click(
            fn=update_bibtex,
            inputs=[citations_data],
            outputs=[bibtex_display]
        )
        example_selector.change(
            fn=load_example_text,
            inputs=[example_selector],
            outputs=[text_input]
        )

if __name__ == "__main__":
    model_path = "../model_v1208/"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path, device)
    # meta_data_path = "../data/corpus_data_arxiv_1129.jsonl"
    meta_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    meta_data = load_meta_data(meta_data_path)
    # citation_map_data_path = "../data/bibtex_info_1202.jsonl"
    citation_map_data_path = "../data/corpus_data_arxiv_1215.jsonl"
    citation_map_data = load_citation_map_data(citation_map_data_path)
    index_dir = "../data/"
    index, lookup_indices = load_faiss_index(index_dir)
    print("index building finished")
    curr_search_candidates = []

    app.queue()
    app.launch(share=True, allowed_paths=["src"])





