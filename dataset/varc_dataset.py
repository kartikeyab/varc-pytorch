import os
import re
import random
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torch

class VARCPairedCanvasTransform:
    def __init__(self, canvas_size=64, max_scale=3):
        self.canvas_size = canvas_size
        self.max_scale = max_scale

    def __call__(self, input_tensor, output_tensor):
        """
        Args:
            input_tensor: (C, H_in, W_in)
            output_tensor: (C, H_out, W_out)
        Returns:
            input_canvas, output_canvas
        """
        # 1. Handle Dimensions
        if input_tensor.dim() == 2: input_tensor = input_tensor.unsqueeze(0)
        if output_tensor.dim() == 2: output_tensor = output_tensor.unsqueeze(0)
        
        in_c, in_h, in_w = input_tensor.shape
        out_c, out_h, out_w = output_tensor.shape

        # 2. Calculate Common Scale Factor
        # We must find a scale that fits BOTH input and output onto the canvas
        max_h = max(in_h, out_h)
        max_w = max(in_w, out_w)
        
        limit_scale = min(self.canvas_size // max_h, self.canvas_size // max_w)
        
        # Prevent scale=0 if image is larger than canvas (unlikely in ARC but safe to handle)
        limit_scale = max(1, limit_scale) 
        
        scale = random.randint(1, min(self.max_scale, limit_scale))
        
        # 3. Calculate Common Position (Top, Left)
        # We calculate the offset based on the LARGEST dimension to ensure both fit
        # Note: This assumes we align them by their top-left corner (standard for ARC)
        final_max_h = max_h * scale
        final_max_w = max_w * scale
        
        max_y_offset = self.canvas_size - final_max_h
        max_x_offset = self.canvas_size - final_max_w
        
        top = random.randint(0, max_y_offset)
        left = random.randint(0, max_x_offset)
        
        # 4. Apply Transform Helper
        def apply_to_single(img, s, t, l):
            # Scale
            if s > 1:
                img = img.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2)
            
            new_h, new_w = img.shape[1:]
            
            # Create Canvas (Background=10)
            canvas = torch.full((img.shape[0], self.canvas_size, self.canvas_size), 
                                fill_value=10, dtype=img.dtype)
            
            # Paste at the specific Top/Left coordinate
            canvas[:, t:t+new_h, l:l+new_w] = img
            return canvas

        input_canvas = apply_to_single(input_tensor, scale, top, left)
        output_canvas = apply_to_single(output_tensor, scale, top, left)
        
        return input_canvas, output_canvas


class VARCDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (str): Path to the dataset root (e.g., 'dataset/train').
            mode (str): 'train' (returns demo pairs) or 'val' (returns infer pairs).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert mode in ['train', 'val'], "Mode must be 'train' or 'val'"
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []
        self.paired_transform = VARCPairedCanvasTransform(canvas_size=64)
        
        # Standard ARC color map (0-9)
        # We don't normalize to [0,1] or [-1,1] typically for ARC; 
        # we keep them as integers 0-9 if using embedding layer for pixels, 
        # or mapped colors if using RGB. 
        # Assuming RGB images saved earlier:
        
        self._scan_directory()

    def _scan_directory(self):
        # Sort tasks to ensure deterministic Task IDs across runs
        task_folders = sorted([
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        
        for task_idx, task_name in enumerate(task_folders):
            task_path = os.path.join(self.root_dir, task_name)
            files = sorted(os.listdir(task_path))
            
            # Filter for the correct pair type based on mode
            prefix = "demo" if self.mode == 'train' else "infer"
            
            # Find all input files matching the prefix (e.g., "demo_0_input.png")
            input_files = [f for f in files if f.startswith(prefix) and f.endswith('_input.png')]
            
            for inp_file in input_files:
                # Construct output filename (e.g., "demo_0_input.png" -> "demo_0_output.png")
                out_file = inp_file.replace('_input.png', '_output.png')
                
                if out_file in files:
                    self.samples.append({
                        'input_path': os.path.join(task_path, inp_file),
                        'output_path': os.path.join(task_path, out_file),
                        'task_id': task_idx,     # Integer ID for the Embedding Layer
                        'task_name': task_name   # String ID for debugging/logging
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Open as RGB (keeping colors consistent)
        # Note: In real ARC models, you often convert RGB -> 0-9 integer grid here.
        # This example assumes standard image tensor output.
        input_img = Image.open(item['input_path']).convert('RGB')
        output_img = Image.open(item['output_path']).convert('RGB')
        

        input_tensor, output_tensor = T.ToTensor()(input_img), T.ToTensor()(output_img) 
        input_img, output_img = self.paired_transform(input_tensor, output_tensor)
            
        return {
            'input': input_img,
            'output': output_img,
            'task_id': item['task_id'],
            'task_name': item['task_name']
        }
