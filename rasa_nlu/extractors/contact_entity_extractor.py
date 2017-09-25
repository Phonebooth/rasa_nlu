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
        _contacts = [
            {
                "entity": "contact",
                "value": contact,
                "start": str(text.find(contact)),
                "end": str(text.find(contact) + len(contact)-1),
                "match": "exact"
            }
            for contact in contacts if contact in text]

        if len(_contacts) == 0:
            _contacts = [
                {
                    "entity": "contact",
                    "value": noun.text,
                    "start": noun.start_char,
                    "end": noun.end_char,
                    "match": "spacy_noun"
                }
                for noun in s_doc.noun_chunks]
        return _contacts
