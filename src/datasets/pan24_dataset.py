import os
import json
import torch
from torch.utils.data import Dataset
from utilities.file_utils import Utils as utils
from transformers import AutoTokenizer
import random

class Pan24Dataset(Dataset):
    def __init__(self, params, split, dataset_name="pan24", is_queries=False, num_sample_per_author=None):
        self.params = params
        self.split = split
        self.dataset_name = dataset_name
        self.num_sample_per_author = num_sample_per_author
        self.is_queries = is_queries
        self.dataset_path = utils.path_exists(os.path.join(self.params.data_dir, split))
        model_folder = self.params.pretrained_model_name_or_path.replace("sentence-transformers/", "")
        local_model_path = os.path.join(utils.transformer_path, model_folder)

        if os.path.exists(local_model_path):
            tokenizer_path = local_model_path
        else:
            tokenizer_path = self.params.pretrained_model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.samples = []
        self.load_pan24_data()

    def reconstruct_authors(self, changes, num_authors):
        """
        Reconstruct paragraph authors:
        - 2 authors: alternate on each 1
        - >2 authors: assign each block of 1 followed by zeros to next author
        """
        paragraph_authors = [1]  # first paragraph always Author 1
        if num_authors == 2:
            current_author = 1
            for c in changes:
                if c == 1:
                    current_author = 3 - current_author  # toggle
                paragraph_authors.append(current_author)
        else:
            current_author = 1
            idx = 0
            while idx < len(changes):
                c = changes[idx]
                if c == 1:
                    current_author += 1
                    if current_author > num_authors:
                        paragraph_authors.append(-1)  # unknown
                        idx += 1
                        continue
                    paragraph_authors.append(current_author)
                    idx += 1
                    while idx < len(changes) and changes[idx] == 0:
                        paragraph_authors.append(current_author)
                        idx += 1
                else:
                    paragraph_authors.append(current_author)
                    idx += 1
        return paragraph_authors

    def load_pan24_data(self):
        for fname in os.listdir(self.dataset_path):
            if not fname.startswith("problem-") or not fname.endswith(".txt"):
                continue

            problem_id = fname.replace("problem-", "").replace(".txt", "")
            text_path = os.path.join(self.dataset_path, fname)
            truth_path = os.path.join(self.dataset_path, f"truth-problem-{problem_id}.json")

            with open(text_path, "r", encoding="utf-8", newline="") as f_txt, \
                open(truth_path, "r", encoding="utf-8") as f_json:

                paragraphs = [line.strip() for line in f_txt if line.strip()]

                truth = json.load(f_json)
                changes = truth["changes"]

                expected_len = len(paragraphs) - 1
                if len(changes) < expected_len:
                    changes += [0] * (expected_len - len(changes))
                elif len(changes) > expected_len:
                    changes = changes[:expected_len]

                num_authors = truth.get("authors", 2)
                paragraph_authors = self.reconstruct_authors(changes, num_authors)

                self.samples.append({
                    "paragraphs": paragraphs,
                    "change_labels": changes,
                    "paragraph-authors": paragraph_authors,
                    "problem_id": problem_id,
                    "authors": num_authors
                })


    def __getitem__(self, idx):
        sample = self.samples[idx]
        paragraphs = sample["paragraphs"]
        paragraph_authors = sample["paragraph-authors"]  # e.g., [1,2]
        changes = sample["change_labels"]               # e.g., [1]
        problem_id = sample["problem_id"]
        num_authors = sample["authors"]

        E = self.params.episode_length
        unique_authors = list(range(1, num_authors + 1))

        # Map paragraphs per author
        author_to_paragraphs = {a: [] for a in unique_authors}
        for p, a in zip(paragraphs, paragraph_authors):
            if a in author_to_paragraphs and a != -1:
                author_to_paragraphs[a].append(p)

        # Ground truth labels per author
        labels_per_author = {a: [] for a in unique_authors}
        for i, change in enumerate(changes):
            prev_author = paragraph_authors[i+1]  # the author who wrote paragraph i
            if prev_author in labels_per_author:
                labels_per_author[prev_author].append(change)

        # Pad/truncate to E-1
        for a in unique_authors:
            labels_per_author[a] = labels_per_author[a][:E-1]
            while len(labels_per_author[a]) < E-1:
                labels_per_author[a].append(-100)

        # Internal labels for visualization (prepend 0 for first paragraph author)
        first_author = paragraph_authors[0]
        internal_labels_per_author = {}
        for a in unique_authors:
            if a == first_author:
                internal_labels_per_author[a] = [0] + labels_per_author[a]
            else:
                internal_labels_per_author[a] = labels_per_author[a][:]

        # Tokenize paragraphs per author
        tokenized_input_ids, tokenized_attention_masks, labels_list = [], [], []
        for a in unique_authors:
            ps = author_to_paragraphs[a][:E]
            while len(ps) < E:
                ps.append("")

            tokenized = self.tokenizer(
                ps,
                padding="max_length",
                truncation=True,
                max_length=self.params.token_max_length,
                return_tensors="pt"
            )
            tokenized_input_ids.append(tokenized["input_ids"])
            tokenized_attention_masks.append(tokenized["attention_mask"])
            labels_list.append(torch.tensor(labels_per_author[a], dtype=torch.long))  # GT only

        input_ids_tensor = torch.stack(tokenized_input_ids)
        attention_mask_tensor = torch.stack(tokenized_attention_masks)
        labels_tensor = torch.stack(labels_list)  # [N, E-1]

        paragraph_level_labels = changes[:E-1]
        while len(paragraph_level_labels) < E-1:
            paragraph_level_labels.append(-100)

        #if getattr(self, "_debug_print_labels_done", False) is False and problem_id == "1":
            #print(f"[DEBUG GETITEM] Problem {problem_id} internal author labels (with prepended 0):")
            #for author_idx, a in enumerate(unique_authors):
            #    print(f"Author {a}: {internal_labels_per_author[a]}")
            #print(f"[DEBUG GETITEM] Problem {problem_id} paragraph-level labels: {paragraph_level_labels}")
            #self._debug_print_labels_done = True

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,  # ground truth per author
            "problem_ids": problem_id,
        }



    def __len__(self):
        return len(self.samples)
