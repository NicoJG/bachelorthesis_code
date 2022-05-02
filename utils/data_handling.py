import torch
import numpy as np


class DataIteratorByEvents:
    def __init__(self, X, y=None, batch_size=1):
        # Iterate through batches of events
        # X must have the event ids of each track in the first column
        # y is optional
        # X (and y) may be a numpy array or a pytorch tensor
        # but internally it is handled as a pytorch tensor
        
        self.batch_size = batch_size
        
        self.is_y_provided = y is not None
        
        if isinstance(X, torch.Tensor):
            self.use_numpy = False
            self.X = X
            if self.is_y_provided:
                assert isinstance(y, torch.Tensor), "If X is a PyTorch Tensor, y also has to be a PyTorch Tensor"
                self.y = y
            else:
                self.y = None
        elif isinstance(X, np.ndarray):
            self.use_numpy = True
            self.X = torch.from_numpy(X)
            if self.is_y_provided:
                assert isinstance(y, np.ndarray), "If X is a numpy array, y also has to be a numpy array"
                self.y = torch.from_numpy(y)
            else:
                self.y = None
        else:
            raise RuntimeError("X is not a PyTorch Tensor and not a numpy array")
        
        # use numpy.unique to find first occurences of each event id
        self.event_ids, self.event_first_idxs = np.unique(self.X[:,0].numpy(), return_index=True)
        # numpy unique sorts the values, so we have to "unsort" them
        unsort_mask = np.argsort(self.event_first_idxs)
        self.event_ids = torch.from_numpy(self.event_ids[unsort_mask])
        self.event_first_idxs = torch.from_numpy(self.event_first_idxs[unsort_mask])
        
        if self.is_y_provided:
            assert self.y.shape[0] == self.event_ids.shape[0], "y must have the same length as unique values in event_ids_by_track!"
        
        self.n_events = self.event_ids.shape[0]
        self.n_tracks = self.X.shape[0]
        self.n_batches = int(np.ceil(self.n_events / self.batch_size))
        
        self.current_event_idx = 0
        
    def __iter__(self):
        self.current_event_idx = 0
        return self
        
    def __next__(self):
        if self.current_event_idx >= self.n_events:
            raise StopIteration
        
        batch_start_event_idx = self.current_event_idx
        batch_stop_event_idx = batch_start_event_idx + self.batch_size # index of the first event that is not in the batch
        
        batch_start_track_idx = self.event_first_idxs[batch_start_event_idx]
        
        if batch_stop_event_idx >= self.n_events:
            # last batch in the dataset
            batch_stop_event_idx = self.n_events
            batch_stop_track_idx = self.n_tracks
        else:
            batch_stop_track_idx = self.event_first_idxs[batch_stop_event_idx]
            
        batch_event_slice = slice(batch_start_event_idx, batch_stop_event_idx)
        batch_track_slice = slice(batch_start_track_idx, batch_stop_track_idx)
        
        self.current_event_idx = batch_stop_event_idx
        
        if self.use_numpy:
            if self.is_y_provided:
                return (self.X[batch_track_slice].numpy(),
                        self.y[batch_event_slice].numpy(),
                        self.event_ids[batch_event_slice].numpy()) 
            else:
                return (self.X[batch_track_slice].numpy(),
                        self.event_ids[batch_event_slice].numpy()) 
                
        if self.is_y_provided:
            return (self.X[batch_track_slice],
                    self.y[batch_event_slice],
                    self.event_ids[batch_event_slice]) 
        else:
            return (self.X[batch_track_slice],
                    self.event_ids[batch_event_slice]) 
        
    def __len__(self):
        return self.n_batches