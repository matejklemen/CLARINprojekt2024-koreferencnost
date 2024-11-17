import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from trankit import Pipeline


# NOTE: currently all entities are of "generic" type
if __name__ == "__main__":
	pipe = Pipeline(lang="slovenian")
	DATA_DIR = "coref149_v1.0"
	TC_NAMESPACE = "{http://www.dspin.de/data/textcorpus}"
	ENTITY_TYPE = "generic"
	KEYS_FOR_UD = ["text", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
	all_files = sorted([fname for fname in os.listdir(DATA_DIR) if fname.endswith(".tcf")],
					   key=lambda _fname: int(_fname.split(".")[-2]))
	print(f"Processing {len(all_files)} files")

	idx_ent = 0
	with open("coref149_corefud.conllu", "w") as f_connlu:
		for idx_file, curr_fname in tqdm(enumerate(all_files)):
			curr_doc = ET.parse(os.path.join(DATA_DIR, curr_fname))
			root = curr_doc.getroot()
			id_doc = curr_fname.split(os.path.sep)[-1]
			print(f"# newdoc id = {id_doc}", file=f_connlu)
			# Following LitBank's CorefUD 1.2 formatting style
			print(f"# global.Entity = eid-etype-head-other", file=f_connlu)
			# Coref149 contains single paragraphs
			print("# newpar", file=f_connlu)

			token_tags = root.findall(f".//{TC_NAMESPACE}token")
			id2tok, id2idx, id2globidx, id2sentidx = {}, {}, {}, {}
			for idx_global, token in enumerate(token_tags):
				id_token = token.attrib["ID"]
				text_token = token.text.strip()

				id2tok[id_token] = text_token
				id2globidx[id_token] = idx_global

			sent_tags = root.findall(f".//{TC_NAMESPACE}sentence")
			words = []
			for idx_sent, sent in enumerate(sent_tags):
				token_ids = sent.attrib["tokenIDs"].split(" ")
				for local_position, _id_tok in enumerate(token_ids):
					id2sentidx[_id_tok] = idx_sent
					id2idx[_id_tok] = local_position
				words.append([id2tok[_id] for _id in token_ids])

			# Automatic annotation with Trankit
			res = pipe(words)
			sents_metadata = [{} for idx_sent in range(len(words))]
			sents_word_data = []
			for idx_sent, sent_data in enumerate(res["sentences"]):
				sents_metadata[idx_sent]["sent_id"] = f"{id_doc}.s{sent_data['id']}"
				sents_metadata[idx_sent]["text"] = " ".join(map(lambda _tok_info: _tok_info["text"], sent_data["tokens"]))
				for idx_tok in range(len(sent_data["tokens"])):
					sent_data["tokens"][idx_tok]["misc"] = []

				sents_word_data.append(sent_data["tokens"])

			# TODO: temporary so that I don't need to run trankit
			# sents_word_data = [[{"misc": []} for _ in range(len(words[_idx_sent]))] for _idx_sent in range(len(words))]

			mentions, clusters = [], []
			for ent in root.findall(f".//{TC_NAMESPACE}entity"):
				id_ent = f"e{idx_ent}"
				idx_ent += 1
				for ref in ent.findall(f"{TC_NAMESPACE}reference"):
					# e.g., t_38 t_39 t_40
					involved_token_ids = sorted(ref.attrib['tokenIDs'].split(" "),
											 key=lambda _tok_id: int(_tok_id.split("_")[-1]))

					# Mention head resolution: see which word in mention has a head outside of the mention (== mention head)
					mention_head = 1
					idx_sent = id2sentidx[involved_token_ids[0]]
					involved_indices = [id2idx[_id] for _id in involved_token_ids]
					involved_indices_set = set(involved_indices)
					# Heads are 1-based (0 = root), convert to 0-based to be compatible with indices
					involved_heads = [sents_word_data[idx_sent][_idx_w]["head"] - 1 for _idx_w in involved_indices]
					for position, _idx_head in enumerate(involved_heads, start=1):
						if _idx_head not in involved_indices_set:
							mention_head = position
							break

					start_token_id = involved_token_ids[0]
					start_idx_sent = id2sentidx[start_token_id]
					start_idx_tok = id2idx[start_token_id]
					sents_word_data[start_idx_sent][start_idx_tok]["misc"].append(f"({id_ent}-{ENTITY_TYPE}-{mention_head}")

					end_token_id = involved_token_ids[-1]
					end_idx_sent = id2sentidx[end_token_id]
					end_idx_tok = id2idx[end_token_id]
					END_TAG = ")" if end_token_id == start_token_id else f"{id_ent})"
					sents_word_data[end_idx_sent][end_idx_tok]["misc"].append(END_TAG)
					# print([id2tok[_id] for _id in involved_token_ids])
					# print(f"{start_idx_tok} - {end_idx_tok}")
					# print("")

			for idx_sent in range(len(sents_word_data)):
				print(f"# sent_id = {sents_metadata[idx_sent]['sent_id']}", file=f_connlu)  # TODO: add full text as a comment
				for idx_word in range(len(sents_word_data[idx_sent])):
					if len(sents_word_data[idx_sent][idx_word]["misc"]) > 0:
						sents_word_data[idx_sent][idx_word]["misc"] = "Entity={}".format("".join(sents_word_data[idx_sent][idx_word]["misc"]))
					else:
						sents_word_data[idx_sent][idx_word]["misc"] = "_"
					print("{}\t{}".format(
						1 + idx_word,
						"\t".join([str(sents_word_data[idx_sent][idx_word].get(_k, "_")) for _k in KEYS_FOR_UD]),
					), file=f_connlu)
				print("", file=f_connlu)
