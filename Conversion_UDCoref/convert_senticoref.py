import re
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import xml.etree.ElementTree as ET

from tqdm import tqdm
from trankit import Pipeline

XML_NAMESPACE = "{http://www.w3.org/XML/1998/namespace}"


def namespace(element):
	# https://stackoverflow.com/a/12946675
	m = re.match(r'\{.*\}', element.tag)
	return m.group(0) if m else ''


def recursively_parse_el(el_tag, opened_ne: str = "O", opened_mentions: list = None) -> Dict:
	eff_opened_mentions = opened_mentions if opened_mentions is not None else []
	id_words, words, lemmas, xposes, feats, ne_tags = [], [], [], [], [], []
	mention_to_id_word = {}

	if el_tag.tag.endswith(("w", "pc")):
		id_word = el_tag.attrib[f"{XML_NAMESPACE}id"]
		word_str = el_tag.text.strip()
		lemma_str = el_tag.attrib["lemma"]
		xpos_str = el_tag.attrib["ana"]
		feats_str = el_tag.attrib["msd"]

		id_words.append(id_word)
		words.append(word_str)
		lemmas.append(lemma_str)
		xposes.append(xpos_str)
		feats.append(feats_str)
		ne_tags.append(opened_ne)

		for _id in eff_opened_mentions:
			_existing = mention_to_id_word.get(_id, [])
			_existing.append(id_word)

			mention_to_id_word[_id] = _existing

	# Named entity or some other type of coreference mention
	elif el_tag.tag.endswith("seg"):
		new_ne = opened_ne
		if el_tag.attrib["type"] == "name":
			assert opened_ne == "O", f"Encountered a nested NE which the script is not designed to handle " \
									 f"({opened_ne}, {el_tag['subtype'].upper()})"
			new_ne = el_tag.attrib["subtype"].upper()
			if new_ne.startswith("DERIV-"):
				new_ne = new_ne[len("DERIV-"):]

		# The mentions can be nested multiple levels, keep track of all mentions at current or shallower level
		id_mention = el_tag.attrib[f"{XML_NAMESPACE}id"]
		_opened_copy = deepcopy(eff_opened_mentions)
		_opened_copy.append(id_mention)

		for _i, _child in enumerate(el_tag):
			_res = recursively_parse_el(_child,
										opened_ne=new_ne,
										opened_mentions=_opened_copy)

			id_words.extend(_res["id_words"])
			words.extend(_res["words"])
			lemmas.extend(_res["lemmas"])
			xposes.extend(_res["xposes"])
			feats.extend(_res["feats"])
			ne_tags.extend(_res["ne_tags"])

			for _id_mention, _id_words in _res["mentions"].items():
				_existing = mention_to_id_word.get(_id_mention, [])
				_existing.extend(_id_words)
				mention_to_id_word[_id_mention] = _existing

		if new_ne != "O":
			ne_tags = [f"B-{_tag}" if _i == 0 else f"I-{_tag}" for _i, _tag in enumerate(ne_tags)]

	else:
		print(f"WARNING: unrecognized tag in recursively_parse_el: {el_tag}. "
			  f"Please open an issue on the HuggingFace datasets repository.")

	return {
		"id_words": id_words, "words": words, "lemmas": lemmas, "xposes": xposes, "feats": feats, "ne_tags": ne_tags,
		"mentions": mention_to_id_word
	}


def parse_sent(sent_tag):
	sent_info = {
		"id_sent": sent_tag.attrib[f"{XML_NAMESPACE}id"],
		"id_words": [], "words": [], "lemmas": [], "xposes": [], "feats": [], "ne_tags": [],
		"mentions": {}
	}

	for el in sent_tag:
		if el.tag.endswith("linkGrp"):
			# Parse coreference clusters later, outside of this function
			continue

		res = recursively_parse_el(el)

		sent_info["id_words"].extend(res["id_words"])
		sent_info["words"].extend(res["words"])
		sent_info["lemmas"].extend(res["lemmas"])
		sent_info["xposes"].extend(res["xposes"])
		sent_info["feats"].extend(res["feats"])
		sent_info["ne_tags"].extend(res["ne_tags"])
		sent_info["mentions"].update(res["mentions"])

	return sent_info


if __name__ == "__main__":
	KEYS_FOR_UD = ["text", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
	file_path = "SUK.TEI/senticoref.xml"
	# Note: entity types are not annotated with coreferences, and cannot be unambiguously propagated from named entities
	ENTITY_TYPE = "generic"
	pipe = Pipeline(lang="slovenian")

	curr_doc = ET.parse(file_path)
	root = curr_doc.getroot()
	NAMESPACE = namespace(root)

	idx_ent = 0
	with open("senticoref_corefud.conllu", "w") as f_connlu:
		for doc in tqdm(root.iterfind(f"{NAMESPACE}div")):
			id_doc = doc.attrib[f"{XML_NAMESPACE}id"]
			print(f"# newdoc id = {id_doc}", file=f_connlu)
			# Following LitBank's CorefUD 1.2 formatting style
			print(f"# global.Entity = eid-etype-head-other", file=f_connlu)

			id2tokinfo = {}
			doc_word_data = []
			doc_mentions = {}
			doc_position = 0
			for idx_par, par in enumerate(doc.findall(f"{NAMESPACE}p")):
				par_words = []
				par_word_data = []

				for idx_sent, sent in enumerate(par.findall(f"{NAMESPACE}s")):
					sent_data = parse_sent(sent)
					sent_words = []
					sent_word_data = []

					for pos_in_sent, (id_token, word_str, lemma_str, xpos_str, feats_str, ne_tag_str) in enumerate(zip(sent_data["id_words"],
																													   sent_data["words"],
																													   sent_data["lemmas"],
																													   sent_data["xposes"],
																													   sent_data["feats"],
																													   sent_data["ne_tags"])):
						sent_words.append(word_str)
						# UPOS, head, and deprel will be obtained using Trankit
						sent_word_data.append({
							"id": id_token, "text": word_str, "lemma": lemma_str,
							"upos": "_", "xpos": xpos_str, "feats": feats_str,
							"head": "_", "deprel": "_", "ne_tag": ne_tag_str,
							"deps": "_", "misc": []
						})

						id2tokinfo[id_token] = {
							"idx_par": idx_par, "idx_sent": idx_sent, "pos_in_sent": pos_in_sent,
							"doc_position": doc_position
						}
						doc_position += 1

					for id_mention, word_ids in sent_data["mentions"].items():
						mention_fmt = {
							"idx_par": idx_par, "idx_sent": idx_sent, "word_indices": [],
							"global_word_indices": []
						}

						for _id in word_ids:
							_info = id2tokinfo[_id]
							mention_fmt["word_indices"].append(_info["pos_in_sent"])
							mention_fmt["global_word_indices"].append(_info["doc_position"])

						doc_mentions[id_mention] = mention_fmt

					par_words.append(sent_words)
					par_word_data.append({
						"id_sent": sent_data["id_sent"],
						"tokens": sent_word_data
					})

				res = pipe(par_words)
				# # TODO: just so that I don't need to run trankit
				# res = {"sentences": [
				# 	{"tokens": [{"upos": "_", "head": "_", "deprel": "_"}
				# 				for _tok in _sent]}
				# ] for _sent in par_words}

				for idx_sent, sent_info in enumerate(res["sentences"]):
					for idx_word, word_info in enumerate(sent_info["tokens"]):
						par_word_data[idx_sent]["tokens"][idx_word]["upos"] = word_info["upos"]
						par_word_data[idx_sent]["tokens"][idx_word]["head"] = word_info["head"]
						par_word_data[idx_sent]["tokens"][idx_word]["deprel"] = word_info["deprel"]

				doc_word_data.append(par_word_data)

			# Preserving order just in case
			unique_clusters = OrderedDict()
			for link_group in doc.findall(f".//{NAMESPACE}linkGrp[@type = 'COREF']"):
				for link in link_group.findall(f"{NAMESPACE}link"):
					# Remove the reference marker ("#") in front of ID
					cluster = tuple(map(lambda _s: _s[1:], link.attrib["target"].split(" ")))
					unique_clusters[cluster] = None

			doc_clusters = []
			for cluster in unique_clusters:
				doc_clusters.append(list(cluster))
				for id_mention in cluster:
					if id_mention not in doc_mentions:
						# Mention is a regular token (id_mention is actually ID of a token)
						_info = id2tokinfo[id_mention]
						doc_mentions[id_mention] = {
							"idx_par": _info["idx_par"], "idx_sent": _info["idx_sent"], "word_indices": [_info["pos_in_sent"]],
							"global_word_indices": [_info["doc_position"]]
						}

			for _cluster in doc_clusters:
				id_ent = f"e{idx_ent}"
				idx_ent += 1

				for _id_mention in _cluster:

					_mention_info = doc_mentions[_id_mention]
					_idx_par = _mention_info["idx_par"]
					_idx_sent = _mention_info["idx_sent"]
					_word_indices = sorted(_mention_info["word_indices"])
					_word_indices_set = set(_word_indices)

					# Mention head resolution: see which word in mention has a head outside of the mention (== mention head)
					mention_head = 1
					_involved_heads = [doc_word_data[_idx_par][_idx_sent]["tokens"][_idx_w]["head"] - 1 for _idx_w in _word_indices]
					for position, _idx_head in enumerate(_involved_heads, start=1):
						if _idx_head not in _word_indices_set:
							mention_head = position
							break

					_start_idx = _word_indices[0]
					doc_word_data[_idx_par][_idx_sent]["tokens"][_start_idx]["misc"].append(f"({id_ent}-{ENTITY_TYPE}-{mention_head}")

					_end_idx = _word_indices[-1]
					END_TAG = ")" if _end_idx == _start_idx else f"{id_ent})"
					doc_word_data[_idx_par][_idx_sent]["tokens"][_end_idx]["misc"].append(END_TAG)

			for par in doc_word_data:
				print("# newpar", file=f_connlu)
				for sent in par:
					print(f"# sent_id = {sent['id_sent']} ", file=f_connlu)
					for _id_w, word_info in enumerate(sent["tokens"], start=1):
						if len(word_info["misc"]) > 0:
							word_info["misc"] = "Entity={}".format("".join(word_info["misc"]))
						else:
							word_info["misc"] = "_"

						print("{}\t{}".format(
							_id_w,
							"\t".join([str(word_info.get(_k, "_")) for _k in KEYS_FOR_UD]),
						), file=f_connlu)
					print("", file=f_connlu)

