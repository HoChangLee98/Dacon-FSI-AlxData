import os
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

class GanProcessor:
    def __init__(
        self,
        subset,
        subset_name:str, 
        column_sdtypes:dict,
        model_name:str = "ctgan",
        n_cls_per_gen:int = 1000, 
        version:str = "baseline"
    ):
        """ 
        Args:
            subset : subset of origianl data
            subset_name : name of fraud type
            column_sdtypes : dict of columns dtypes
            model_name (str, optional): choose generating method [ctgan, tvae]. Defaults to "ctgan".
            n_cls_per_gen (int, optional): number of generated data. Defaults to 1000.
        """
        self.subset = subset
        self.subset_name = subset_name
        self.column_sdtypes = column_sdtypes        
        self.model_name = model_name
        self.n_cls_per_gen = n_cls_per_gen
        self.version = version
        
        self.metadata = SingleTableMetadata()                
        self.metadata.detect_from_dataframe(self.subset)
        self.metadata.set_primary_key(None)
        

    def synthesizer(
        self,
        epochs:int = None
    ):
        if self.model_name == "ctgan":
            synthesizer_ = CTGANSynthesizer(self.metadata, epochs=epochs)
        elif self.model_name == "tvae":
            synthesizer_ = TVAESynthesizer(self.metadata, epochs=epochs)
        
        return synthesizer_
    

    def _save(
        self,
        epochs:int = None
    ):
        for column, sdtype in self.column_sdtypes.items():
            self.metadata.update_column(column_name=column, sdtype=sdtype)

        synthesizer_ = self.synthesizer(epochs=epochs)
        synthesizer_.fit(self.subset)

        directory = f"./{self.model_name}/{self.version}_synthesizer"
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        synthesizer_.save(f"{directory}/{self.subset_name}_synthesizer.pkl")
        