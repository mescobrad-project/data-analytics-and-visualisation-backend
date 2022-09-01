from pydantic import BaseModel


# Base classthat contains information for all other classess needed in every call
# Currently not used
class ModelBase(BaseModel):
    file_name:  str


class ModelMNENotebookConfiguration(BaseModel):
    bipolar_references: list
    # bipolar_anode: str
    # bipolar_cathode: str

    average_channel: str

    notches_enabled: bool
    notches_length: str
    selection_channel: str
    selection_start_time: str
    end_time: str

class ModelSelectionChannelReference(BaseModel):
    selection_channel: str
    selection_start_time: str
    end_time: str
