from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class ContactEntityExtractor(EntityExtractor):
    name = "ner_contact"

    provides = ["entities"]

    requires = ["spacy_doc"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message.text, message.get("contacts", []), message.get("spacy_doc")))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, text, contacts, s_doc):
        # type: (Doc) -> List[Dict[Text, Any]]
        _contacts = []
        for span in s_doc.noun_chunks:
            _contacts.append(
            {
                "entity": "contact",
                "value": span.text,
                "start": span.start_char,
                "end": span.end_char,
                "match": "spacy_noun"
            })

        _exact_contacts = [
            {
                "entity": "contact",
                "value": contact,
                "start": str(text.find(contact)),
                "end": str(text.find(contact)+len(contact)),
                "match": "exact"
            }
            for contact in contacts if contact in text]

        _contacts = self.merge_contacts(_contacts, _exact_contacts, 'value')

        # take the maximum length contact assuming that will be the most precise
        # match "Drew" vs. "Drew G's cell"
        try:
            _contacts = [max(_contacts, key=lambda c: len(c['value']))]
        except ValueError:
            _contacts = []
        return _contacts

    def merge_contacts(self, list1, list2, key):
        merged = {}
        for item in list1+list2:
            if item[key] in merged:
                merged[item[key]].update(item)
            else:
                merged[item[key]] = item
        return merged.values()
