import os

from tqdm import tqdm
from trankit import Pipeline

if __name__ == "__main__":
	DATA_DIR = "senticoref_private"
	pipe = Pipeline(lang="slovenian")
	all_files = sorted([fname for fname in os.listdir(DATA_DIR) if fname.endswith(".tsv")],
					   key=lambda _fname: int(_fname.split(".")[0]))

	idx_ent = 0
	with open("senticoref_private_corefud_unlabeled.conllu", "w") as f_connlu:
		for fname in tqdm(all_files, total=len(all_files)):
			print(f"File '{fname}'")
			fpath = os.path.join(DATA_DIR, fname)

			KEYS_FOR_UD = ["text", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
			IGNORE_LINESTARTS = ["#FORMAT", "#T_SP", "#T_CH"]
			id_doc = fname
			print(f"# newdoc id = {id_doc}", file=f_connlu)
			# Following LitBank's CorefUD 1.2 formatting style
			print(f"# global.Entity = eid-etype-head-other", file=f_connlu)
			print("# newpar", file=f_connlu)

			with open(fpath) as f:
				lines = list(map(lambda _s: _s.strip(), f.readlines()))
				lines.append("")  # Append an empty line at the end so that the code can be more consistent

			# Skip until the first sentence data
			while True:
				first_line = lines[0]

				if first_line.startswith("#Text"):
					break
				lines = lines[1:]
			# ----------------------------------

			sent_words, par_words = [], []
			idx_sent, idx_word = 0, 0
			doc_mentions, doc_entities = {}, {}

			for line in lines:
				if len(line) == 0:
					par_words.append(sent_words)
					sent_words = []
					idx_sent += 1
					idx_word = 0
					continue

				elif line.startswith("#"):  # Text=...
					continue

				parts = line.split("\t")
				text = parts[2]
				sent_words.append(text)

				coref_anns = parts[-2]  # e.g., *->72-7
				if coref_anns != "_":
					for ann in coref_anns.split("|"):
						_, id_mention = ann.split("->")
						id_ent, id_mention = id_mention.split("-")

						# Assign a corpus-level unique entity ID to a document-level unique entity ID
						if id_ent not in doc_entities:
							doc_entities[id_ent] = f"e{idx_ent}"
							idx_ent += 1
						# Obtain the corpus-level unique entity ID
						id_ent = doc_entities[id_ent]

						# Reconstruct <entity_id>-<mention_id> format
						id_mention = f"{id_ent}-{id_mention}"
						existing_mention_data = doc_mentions.get(id_mention, {})

						existing_mention_data["idx_sent"] = idx_sent
						existing_word_indices = existing_mention_data.get("word_indices", [])
						existing_word_indices.append(idx_word)
						existing_mention_data["word_indices"] = existing_word_indices
						doc_mentions[id_mention] = existing_mention_data

				idx_word += 1

			res = pipe(par_words)
			for sent_info in res["sentences"]:
				for word_info in sent_info["tokens"]:
					word_info["deps"] = "_"
					word_info["misc"] = []

			for _id_mention, _mention_data in doc_mentions.items():
				_idx_sent = _mention_data["idx_sent"]
				_word_indices = sorted(_mention_data["word_indices"])

				_id_ent = _id_mention.split("-")[0]
				_start_idx = _word_indices[0]
				_end_idx = _word_indices[-1]
				END_TAG = ")" if _end_idx == _start_idx else f"{_id_ent})"
				# res["sentences"][_idx_sent]["tokens"][_start_idx]["misc"].append(f"({_id_ent}")
				# res["sentences"][_idx_sent]["tokens"][_end_idx]["misc"].append(END_TAG)

			for idx_sent_print, sent_info in enumerate(res["sentences"]):
				print(f"# sent_id = {id_doc}.{idx_sent_print}", file=f_connlu)
				for _id_w, word_info in enumerate(sent_info["tokens"], start=1):
					if len(word_info["misc"]) > 0:
						word_info["misc"] = "Entity={}".format("".join(word_info["misc"]))
					else:
						word_info["misc"] = "_"

					print("{}\t{}".format(
						_id_w,
						"\t".join([str(word_info.get(_k, "_")) for _k in KEYS_FOR_UD]),
					), file=f_connlu)
				print("", file=f_connlu)
