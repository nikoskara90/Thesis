import os
import json
import torch
from torch.utils.data import Dataset
from utilities.file_utils import Utils as utils
from transformers import AutoTokenizer

class Pan21Dataset(Dataset):
    def __init__(self, params, split, dataset_name="pan21", is_queries=False, num_sample_per_author=None):
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
        self.load_pan21_data()

    def load_pan21_data(self):
        for fname in os.listdir(self.dataset_path):
            if not fname.startswith("problem-") or not fname.endswith(".txt"):
                continue

            problem_id = fname.replace("problem-", "").replace(".txt", "")
            text_path = os.path.join(self.dataset_path, fname)
            truth_path = os.path.join(self.dataset_path, f"truth-problem-{problem_id}.json")

            with open(text_path, "r", encoding="utf-8") as f_txt, open(truth_path, "r", encoding="utf-8") as f_json:
                full_text = f_txt.read().strip()
                paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
                if len(paragraphs) <= 1:
                    paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]
                truth = json.load(f_json)
                changes = truth["changes"]

                assert len(paragraphs) == len(truth["paragraph-authors"])
                assert len(changes) == len(paragraphs) - 1

                self.samples.append({
                    "paragraphs": paragraphs,
                    "change_labels": changes,
                    "paragraph-authors": truth["paragraph-authors"], 
                    "problem_id": problem_id,
                    "site": truth.get("site", "")
                })

    def __getitem__(self, idx):
        sample = self.samples[idx]
        paragraphs = sample["paragraphs"]
        authors = sample["paragraph-authors"]
        changes = sample["change_labels"]
        problem_id = sample["problem_id"]

        E = self.params.episode_length
        unique_authors = sorted(set(authors))
        N = max(len(set(s["paragraph-authors"])) for s in self.samples)

        # Map paragraphs by author
        author_to_paragraphs = {a: [] for a in unique_authors}
        for p, a in zip(paragraphs, authors):
            author_to_paragraphs[a].append(p)

        # Author-level labels
        labels_per_author = {a: [] for a in unique_authors}
        for i in range(1, len(paragraphs)):
            curr_author = authors[i]
            labels_per_author[curr_author].append(changes[i - 1])

        # Truncate/pad each author to E-1
        for a in unique_authors:
            labels_per_author[a] = labels_per_author[a][:E-1]
            while len(labels_per_author[a]) < E-1:
                labels_per_author[a].append(-100)

        # Tokenize per author
        tokenized_input_ids, tokenized_attention_masks = [], []
        labels_list = []
        for a in unique_authors:
            ps = author_to_paragraphs[a][:E]
            while len(ps) < E:
                ps.append("")  # pad paragraphs

            tokenized = self.tokenizer(
                ps,
                padding="max_length",
                truncation=True,
                max_length=self.params.token_max_length,
                return_tensors="pt"
            )
            tokenized_input_ids.append(tokenized["input_ids"])
            tokenized_attention_masks.append(tokenized["attention_mask"])
            labels_list.append(torch.tensor(labels_per_author[a], dtype=torch.long))

        # Pad authors if fewer than N
        while len(tokenized_input_ids) < N:
            tokenized_input_ids.append(torch.zeros(E, self.params.token_max_length, dtype=torch.long))
            tokenized_attention_masks.append(torch.zeros(E, self.params.token_max_length, dtype=torch.long))
            labels_list.append(torch.full((E-1,), -100, dtype=torch.long))

        input_ids_tensor = torch.stack(tokenized_input_ids)          # [N, E, L]
        attention_mask_tensor = torch.stack(tokenized_attention_masks)  # [N, E, L]
        labels_tensor = torch.stack(labels_list)                      # [N, E-1]

        # Paragraph-level labels (sequence of all paragraph transitions)
        paragraph_level_labels = changes[:E-1]
        while len(paragraph_level_labels) < E-1:
            paragraph_level_labels.append(-100)

        # Debug prints
        #if getattr(self, "_debug_print_labels_done", False) is False and problem_id == "1":
         #   print(f"[DEBUG GETITEM] Problem {problem_id} author-level labels:")
          #  for author_idx, lbls in enumerate(labels_tensor):
           #     print(f"Author {author_idx + 1}: [{','.join(map(str, lbls.tolist()))}]")

            #print(f"[DEBUG GETITEM] Problem {problem_id} paragraph-level labels: [{','.join(map(str, paragraph_level_labels))}]")
            #self._debug_print_labels_done = True

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
            "problem_ids": problem_id
        }


    def __len__(self):
        return len(self.samples)
