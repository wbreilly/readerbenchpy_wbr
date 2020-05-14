from typing import Dict, List

from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.text_element_type import TextElementType
from rb.core.cscl.cscl_indices import CsclIndices
from rb.utils.rblogger import Logger


class Participant:

	def __init__(self, participant_id: str):
		self.participant_id = participant_id
		self.indices = dict()

	def get_id(self) -> str:
		return self.participant_id

	def get_index(self, index: str):
		if index in self.indices:
			return self.indices[index]

		return 0

	def set_index(self, index: str, value: float):
		self.indices[index] = value