class DiskLazyTimeSeriesSequence(collections.abc.Sequence):
    def __init__(self, data_dir, group_keys, time_col, value_cols, static_cols=None, freq='D'):
        self.data_dir = data_dir
        self.group_keys = group_keys  # A simple list of your unique group IDs
        self.time_col = time_col
        self.value_cols = value_cols
        self.static_cols = static_cols
        self.freq = freq

    def __len__(self):
        return len(self.group_keys)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
            
        group_key = self.group_keys[idx]
        
        # Read exactly one tiny group file off the hard drive
        group_df = pd.read_parquet(os.path.join(self.data_dir, f"group_{group_key}.parquet"))
        
        static_df = None
        if self.static_cols:
            static_df = group_df[self.static_cols].iloc[[0]].reset_index(drop=True)
            
        return TimeSeries.from_dataframe(
            df=group_df,
            time_col=self.time_col,
            value_cols=self.value_cols,
            static_covariates=static_df,
            freq=self.freq,
            fill_missing_dates=False
        )

