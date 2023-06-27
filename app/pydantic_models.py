from pydantic import BaseModel


# Base classthat contains information for all other classess needed in every call
# Currently not used
class ModelBase(BaseModel):
    file_name:  str


class ModelNotebookAndSelectionConfiguration(BaseModel):
    # Bipolar References
    bipolar_references: list

    # Type of reference
    type_of_reference: str
    channels_reference: list

    # Notches
    notches_enabled: bool
    notches_length: str

    # Selection of Part
    selection_channel: str
    selection_start_time: str
    selection_end_time: str
    repairing_artifacts_ica: bool
    n_components: str
    list_exclude_ica: list
    ica_method: str


class ModelSelectionChannelReference(BaseModel):
    selection_channel: str
    selection_start_time: str
    end_time: str

# class ModelMultipleComparisons(BaseModel):
#     workflow_id: str
#     step_id: str
#     run_id: str
#     p_value: list
#     method: str
#     alpha: float
