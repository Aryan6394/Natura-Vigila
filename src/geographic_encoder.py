"""
Geographic region encoder for species distribution data.
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from config.model_config import GEOGRAPHIC_REGIONS


class GeographicEncoder:
    """Encoder for geographic region data."""
    
    def __init__(self):
        self.regions = GEOGRAPHIC_REGIONS
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        self.idx_to_region = {idx: region for idx, region in enumerate(self.regions)}
    
    def encode(self, regions):
        """
        Encode a list of regions to a multi-hot vector.
        
        Args:
            regions: List of region names
            
        Returns:
            Multi-hot encoded vector
        """
        vector = np.zeros(len(self.regions))
        
        if isinstance(regions, str):
            regions = [regions]
        
        for region in regions:
            if region in self.region_to_idx:
                idx = self.region_to_idx[region]
                vector[idx] = 1
        
        return vector
    
    def decode(self, vector, threshold=0.5):
        """
        Decode a multi-hot vector to a list of regions.
        
        Args:
            vector: Multi-hot encoded vector
            threshold: Threshold for binary classification
            
        Returns:
            List of region names
        """
        regions = []
        for idx, value in enumerate(vector):
            if value >= threshold:
                regions.append(self.idx_to_region[idx])
        
        return regions
    
    def get_region_distribution(self, data):
        """
        Get distribution of species across regions.
        
        Args:
            data: DataFrame or list of region assignments
            
        Returns:
            Dictionary with region counts
        """
        distribution = {region: 0 for region in self.regions}
        
        if isinstance(data, pd.DataFrame):
            # Assume 'regions' column exists
            for regions in data['regions']:
                if isinstance(regions, str):
                    regions = [regions]
                for region in regions:
                    if region in distribution:
                        distribution[region] += 1
        else:
            # Assume list of region lists
            for regions in data:
                if isinstance(regions, str):
                    regions = [regions]
                for region in regions:
                    if region in distribution:
                        distribution[region] += 1
        
        return distribution


def create_region_mapping(taxonomy_data, iucn_data=None):
    """
    Create mapping between species and their geographic regions.
    
    Args:
        taxonomy_data: DataFrame with species taxonomy
        iucn_data: Optional IUCN spatial data
        
    Returns:
        DataFrame with species and their regions
    """
    # Placeholder implementation
    # In reality, this would parse IUCN spatial data or other sources
    
    region_mapping = []
    
    for idx, row in taxonomy_data.iterrows():
        species_name = row.get('sciName', 'Unknown')
        
        # Placeholder: assign random regions for demo
        # In production, use actual geographic data
        regions = ['Africa']  # Default
        
        region_mapping.append({
            'species': species_name,
            'regions': regions
        })
    
    return pd.DataFrame(region_mapping)


def visualize_region_distribution(distribution, save_path=None):
    """
    Visualize species distribution across geographic regions.
    
    Args:
        distribution: Dictionary with region counts
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    
    regions = list(distribution.keys())
    counts = list(distribution.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(regions, counts, color='steelblue')
    plt.xlabel('Geographic Region')
    plt.ylabel('Number of Species')
    plt.title('Species Distribution Across Geographic Regions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Test encoder
    encoder = GeographicEncoder()
    
    # Test encoding
    regions = ["Africa", "Asia"]
    encoded = encoder.encode(regions)
    print(f"Encoded: {regions} -> {encoded}")
    
    # Test decoding
    decoded = encoder.decode(encoded)
    print(f"Decoded: {encoded} -> {decoded}")
    
    # Test distribution
    sample_data = [
        ["Africa", "Asia"],
        ["Europe"],
        ["Africa"],
        ["North America", "South America"],
    ]
    distribution = encoder.get_region_distribution(sample_data)
    print(f"\nRegion distribution: {distribution}")
