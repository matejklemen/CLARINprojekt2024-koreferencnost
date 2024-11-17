""" Slovene corpus for coreference resolution coref149. """


import os
import xml.etree.ElementTree as ET
import datasets


_CITATION = """\
@article{coref149, 
	author={Žitnik, Slavko and Bajec, Marko},
	title={Odkrivanje koreferenčnosti v slovenskem jeziku na označenih besedilih iz coref149},
	journal={Slovenščina 2.0: empirične, aplikativne in interdisciplinarne raziskave},
	number={1},
	volume={6},
	year={2018},
	month={Jun.},
	pages={37–67},
	doi={10.4312/slo2.0.2018.1.37-67}
}
"""

_DESCRIPTION = """\
Slovene corpus for coreference resolution. Contains manually annotated coreferences.
"""

_HOMEPAGE = "http://hdl.handle.net/11356/1182"

_LICENSE = "Creative Commons - Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"

_URLS = {
	"coref149": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1182/coref149_v1.0.zip"
}


class Coref149(datasets.GeneratorBasedBuilder):
	"""Slovene corpus for coreference resolution."""

	VERSION = datasets.Version("1.0.0")

	def _info(self):
		features = datasets.Features(
			{
				"id_doc": datasets.Value("string"),
				"words": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
				"mentions": [{
					"id_mention": datasets.Value("string"),
					"mention_data": {
						"idx_sent": datasets.Value("uint32"),
						"word_indices": datasets.Sequence(datasets.Value("uint32")),
						"global_word_indices": datasets.Sequence(datasets.Value("uint32"))
					}
				}],
				"coref_clusters": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
			}
		)

		return datasets.DatasetInfo(
			description=_DESCRIPTION,
			features=features,
			homepage=_HOMEPAGE,
			license=_LICENSE,
			citation=_CITATION,
		)

	def _split_generators(self, dl_manager):
		urls = _URLS["coref149"]
		data_dir = dl_manager.download_and_extract(urls)
		return [
			datasets.SplitGenerator(
				name=datasets.Split.TRAIN,
				gen_kwargs={
					"data_dir": data_dir
				}
			)
		]

	def _generate_examples(self, data_dir):
		TC_NAMESPACE = "{http://www.dspin.de/data/textcorpus}"
		all_files = sorted([fname for fname in os.listdir(data_dir) if fname.endswith(".tcf")],
						   key=lambda _fname: int(_fname.split(".")[-2]))

		for idx_file, curr_fname in enumerate(all_files):
			curr_doc = ET.parse(os.path.join(data_dir, curr_fname))
			root = curr_doc.getroot()
			id_doc = curr_fname.split(os.path.sep)[-1]

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

			mentions, clusters = [], []
			for ent in root.findall(f".//{TC_NAMESPACE}entity"):
				curr_cluster = []
				for ref in ent.findall(f"{TC_NAMESPACE}reference"):
					id_mention = f"{id_doc}.{ref.attrib['ID']}"
					curr_cluster.append(id_mention)
					curr_mention = {
						"id_mention": id_mention,
						"mention_data": {
							"idx_sent": None,
							"word_indices": [],
							"global_word_indices": []
						}
					}

					for id_token in ref.attrib['tokenIDs'].split(" "):
						curr_mention["mention_data"]["idx_sent"] = id2sentidx[id_token]
						curr_mention["mention_data"]["word_indices"].append(id2idx[id_token])
						curr_mention["mention_data"]["global_word_indices"].append(id2globidx[id_token])

					mentions.append(curr_mention)

				clusters.append(curr_cluster)

			yield idx_file, {
				"id_doc": id_doc,
				"words": words,
				"mentions": mentions,
				"coref_clusters": clusters
			}
