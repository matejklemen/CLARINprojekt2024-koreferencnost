""" Slovene corpus for coreference resolution. """
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import datasets
import xml.etree.ElementTree as ET
import re


_CITATION = """\
@misc{suk,
    title = {Training corpus {SUK} 1.1},
    author = {Arhar Holdt, {\v S}pela and Krek, Simon and Dobrovoljc, Kaja and Erjavec, Toma{\v z} and Gantar, Polona and {\v C}ibej, Jaka and Pori, Eva and Ter{\v c}on, Luka and Munda, Tina and {\v Z}itnik, Slavko and Robida, Nejc and Blagus, Neli and Mo{\v z}e, Sara and Ledinek, Nina and Holz, Nanika and Zupan, Katja and Kuzman, Taja and Kav{\v c}i{\v c}, Teja and {\v S}krjanec, Iza and Marko, Dafne and Jezer{\v s}ek, Lucija and Zajc, Anja},
    url = {http://hdl.handle.net/11356/1959},
    note = {Slovenian language resource repository {CLARIN}.{SI}},
    year = {2024}
}
"""

_DESCRIPTION = """\
Slovene corpus for coreference resolution. Contains automatically(?) annotated named entities, manually annotated 
coreferences, and manually verified lemmas and morphosyntactic tags.
"""

_HOMEPAGE = "http://hdl.handle.net/11356/1959"

_LICENSE = "Creative Commons - Attribution-{ShareAlike} 4.0 International ({CC} {BY}-{SA} 4.0)"

_URLS = {
    "suk.tei": "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1959/SUK.TEI.zip",
}


XML_NAMESPACE = "{http://www.w3.org/XML/1998/namespace}"


def namespace(element):
    # https://stackoverflow.com/a/12946675
    m = re.match(r'\{.*\}', element.tag)
    return m.group(0) if m else ''


def recursively_parse_el(el_tag, opened_ne: str = "O", opened_mentions: list = None) -> Dict:
    """
    :param el_tag: XML ETree tag
    :param opened_ne: Named entity tag encountered at the previous level(s) of the recursive parse
    :param opened_mentions: IDs of mentions encountered at the previous level(s) of the recursive parse.
        The word in the current tag is part of these mentions.
    """
    eff_opened_mentions = opened_mentions if opened_mentions is not None else []
    id_words, words, lemmas, msds, ne_tags = [], [], [], [], []
    mention_to_id_word = {}

    if el_tag.tag.endswith(("w", "pc")):
        id_word = el_tag.attrib[f"{XML_NAMESPACE}id"]
        word_str = el_tag.text.strip()
        lemma_str = el_tag.attrib["lemma"]
        msd_str = el_tag.attrib["ana"]

        id_words.append(id_word)
        words.append(word_str)
        lemmas.append(lemma_str)
        msds.append(msd_str)
        ne_tags.append(opened_ne)

        for _id in eff_opened_mentions:
            _existing = mention_to_id_word.get(_id, [])
            _existing.append(id_word)

            mention_to_id_word[_id] = _existing

    # Named entity or some other type of coreference mention
    elif el_tag.tag.endswith("seg"):
        new_ne = opened_ne
        if el_tag.attrib["type"] == "name":
            assert opened_ne == "O", f"Potentially encountered a nested NE ({opened_ne}, {el_tag['subtype'].upper()})"
            new_ne = el_tag.attrib["subtype"].upper()

            # Discard information about derived named entities
            if new_ne.startswith("DERIV-"):
                new_ne = new_ne[len("DERIV-"):]

        # The mentions can be nested multiple levels, keep track of all mentions at current or shallower level
        id_mention = el_tag.attrib[f"{XML_NAMESPACE}id"]
        _opened_copy = deepcopy(eff_opened_mentions)
        _opened_copy.append(id_mention)

        for _i, _child in enumerate(el_tag):
            _res = recursively_parse_el(_child, opened_ne=new_ne, opened_mentions=_opened_copy)

            id_words.extend(_res["id_words"])
            words.extend(_res["words"])
            lemmas.extend(_res["lemmas"])
            msds.extend(_res["msds"])
            ne_tags.extend(_res["ne_tags"])

            for _id_mention, _id_words in _res["mentions"].items():
                _existing = mention_to_id_word.get(_id_mention, [])
                _existing.extend(_id_words)
                mention_to_id_word[_id_mention] = _existing

        if new_ne != "O":  # IOB2
            ne_tags = [f"B-{_tag}" if _i == 0 else f"I-{_tag}" for _i, _tag in enumerate(ne_tags)]

    else:
        print(f"WARNING: unrecognized tag in `recursively_parse_el`: {el_tag}. "
              f"Please open an issue on the HuggingFace datasets repository.")

    return {
        "id_words": id_words, "words": words, "lemmas": lemmas, "msds": msds, "ne_tags": ne_tags,
        "mentions": mention_to_id_word
    }


def parse_sent(sent_tag):
    sent_info = {
        "id_words": [], "words": [], "lemmas": [], "msds": [], "ne_tags": [],
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
        sent_info["msds"].extend(res["msds"])
        sent_info["ne_tags"].extend(res["ne_tags"])
        sent_info["mentions"].update(res["mentions"])

    return sent_info


class SentiCoref(datasets.GeneratorBasedBuilder):
    """Slovene corpus for coreference resolution."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "id_doc": datasets.Value("string"),
                "words": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("string")))),
                "lemmas": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("string")))),
                "msds": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("string")))),
                "ne_tags": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("string")))),
                "mentions": [{
                    "id_mention": datasets.Value("string"),
                    "mention_data": {
                        "idx_par": datasets.Value("uint32"),
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
        urls = _URLS["suk.tei"]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_path": os.path.join(data_dir, "SUK.TEI", "senticoref.xml")}
            )
        ]

    def _generate_examples(self, file_path):
        curr_doc = ET.parse(file_path)
        root = curr_doc.getroot()
        NAMESPACE = namespace(root)

        for idx_doc, doc in enumerate(root.iterfind(f"{NAMESPACE}div")):
            id2tokinfo = {}

            doc_words, doc_lemmas, doc_msds, doc_ne_tags = [], [], [], []
            doc_mentions = {}
            doc_position = 0

            # Step 1: Extract everything but the coreference clusters
            # Clusters are marked at sentence level so they are often duplicated - find unique clusters afterwards
            for idx_par, par in enumerate(doc.findall(f"{NAMESPACE}p")):
                par_words, par_lemmas, par_msds, par_ne_tags = [], [], [], []

                for idx_sent, sent in enumerate(par.findall(f"{NAMESPACE}s")):
                    sent_data = parse_sent(sent)

                    par_words.append(sent_data["words"])
                    par_lemmas.append(sent_data["lemmas"])
                    par_msds.append(sent_data["msds"])
                    par_ne_tags.append(sent_data["ne_tags"])

                    for pos_in_sent, (id_token, word_str, lemma_str, msd_str) in enumerate(zip(sent_data["id_words"],
                                                                                               sent_data["words"],
                                                                                               sent_data["lemmas"],
                                                                                               sent_data["msds"])):
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

                doc_words.append(par_words)
                doc_lemmas.append(par_lemmas)
                doc_msds.append(par_msds)
                doc_ne_tags.append(par_ne_tags)

            # Step 2: extract coreference clusters
            unique_clusters = OrderedDict()  # Preserving order just in case
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
                        # Mention may be a regular token, i.e. a word referring to an entity
                        # (`id_mention` is then the ID of a token)
                        _info = id2tokinfo[id_mention]
                        doc_mentions[id_mention] = {
                            "idx_par": _info["idx_par"], "idx_sent": _info["idx_sent"],
                            "word_indices": [_info["pos_in_sent"]],
                            "global_word_indices": [_info["doc_position"]]
                        }

            # Convert to list of dictionaries as datasets expects fixed key names
            doc_mentions_list = []
            for id_mention, mention_data in doc_mentions.items():
                doc_mentions_list.append({
                    "id_mention": id_mention,
                    "mention_data": mention_data
                })

            yield idx_doc, {
                "id_doc": doc.attrib[f"{XML_NAMESPACE}id"],
                "words": doc_words, "lemmas": doc_lemmas, "msds": doc_msds, "ne_tags": doc_ne_tags,
                "mentions": doc_mentions_list,
                "coref_clusters": doc_clusters
            }
