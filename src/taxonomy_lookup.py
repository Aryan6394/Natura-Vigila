"""
Taxonomy lookup utilities for species information.
"""

import pandas as pd
import sys
sys.path.append('..')
from config.model_config import TAXONOMY_FIELDS, DATA_PATHS


def load_taxonomy_table(filepath=None):
    """Load the taxonomy table from CSV."""
    if filepath is None:
        filepath = DATA_PATHS['taxonomy']
    
    try:
        taxonomy_df = pd.read_csv(filepath)
        return taxonomy_df
    except FileNotFoundError:
        print(f"Taxonomy table not found at {filepath}")
        print("Creating placeholder taxonomy table...")
        return create_placeholder_taxonomy_table()


def create_placeholder_taxonomy_table():
    """Create a placeholder taxonomy table for demo purposes."""
    taxonomy_data = {
        'kingdom': ['Animalia'] * 6,
        'phylum': ['Chordata'] * 6,
        'class': ['Mammalia', 'Mammalia', 'Aves', 'Reptilia', 'Mammalia', 'Mammalia'],
        'order': ['Primates', 'Carnivora', 'Sphenisciformes', 'Testudines', 'Proboscidea', 'Artiodactyla'],
        'family': ['Hominidae', 'Ursidae', 'Spheniscidae', 'Cheloniidae', 'Elephantidae', 'Rhinocerotidae'],
        'genus': ['Pongo', 'Ursus', 'Spheniscus', 'Chelonia', 'Loxodonta', 'Diceros'],
        'species': ['abelii', 'maritimus', 'humboldti', 'mydas', 'africana', 'bicornis'],
        'sciName': ['Pongo abelii', 'Ursus maritimus', 'Spheniscus humboldti', 
                   'Chelonia mydas', 'Loxodonta africana', 'Diceros bicornis'],
        'conservation_status': ['CR', 'VU', 'VU', 'EN', 'EN', 'CR'],
        'common_name': ['Sumatran Orangutan', 'Polar Bear', 'Humboldt Penguin', 
                       'Green Sea Turtle', 'African Elephant', 'Black Rhinoceros']
    }
    
    return pd.DataFrame(taxonomy_data)


def get_taxonomy_info(species_name=None, conservation_status=None):
    """
    Get taxonomy information for a species.
    
    Args:
        species_name: Scientific name of the species
        conservation_status: Conservation status (if species_name not provided)
        
    Returns:
        Dictionary with taxonomy fields
    """
    taxonomy_df = load_taxonomy_table()
    
    if species_name:
        # Look up by scientific name
        match = taxonomy_df[taxonomy_df['sciName'] == species_name]
    elif conservation_status:
        # Get first match for conservation status (for demo)
        match = taxonomy_df[taxonomy_df['conservation_status'] == conservation_status]
    else:
        # Return first entry as default
        match = taxonomy_df.head(1)
    
    if match.empty:
        # Return placeholder if no match found
        return {
            'kingdom': 'Animalia',
            'phylum': 'Chordata',
            'class': 'Unknown',
            'order': 'Unknown',
            'family': 'Unknown',
            'genus': 'Unknown',
            'species': 'Unknown',
            'sciName': 'Unknown species'
        }
    
    # Extract taxonomy fields
    taxonomy_info = {}
    for field in TAXONOMY_FIELDS:
        if field in match.columns:
            taxonomy_info[field] = match.iloc[0][field]
        else:
            taxonomy_info[field] = 'Unknown'
    
    return taxonomy_info


def search_species(query, field='sciName'):
    """
    Search for species in the taxonomy table.
    
    Args:
        query: Search query string
        field: Field to search in (default: sciName)
        
    Returns:
        DataFrame with matching species
    """
    taxonomy_df = load_taxonomy_table()
    
    if field not in taxonomy_df.columns:
        print(f"Field '{field}' not found in taxonomy table")
        return pd.DataFrame()
    
    # Case-insensitive search
    mask = taxonomy_df[field].str.contains(query, case=False, na=False)
    results = taxonomy_df[mask]
    
    return results


def get_species_by_conservation_status(status):
    """Get all species with a given conservation status."""
    taxonomy_df = load_taxonomy_table()
    
    if 'conservation_status' not in taxonomy_df.columns:
        print("Conservation status field not found in taxonomy table")
        return pd.DataFrame()
    
    results = taxonomy_df[taxonomy_df['conservation_status'] == status]
    return results


def save_taxonomy_table(taxonomy_df, filepath=None):
    """Save taxonomy table to CSV."""
    if filepath is None:
        filepath = DATA_PATHS['taxonomy']
    
    taxonomy_df.to_csv(filepath, index=False)
    print(f"Taxonomy table saved to {filepath}")


if __name__ == "__main__":
    # Create and save placeholder taxonomy table
    taxonomy_df = create_placeholder_taxonomy_table()
    print("Placeholder Taxonomy Table:")
    print(taxonomy_df)
    
    # Test taxonomy lookup
    print("\nLooking up taxonomy for Pongo abelii:")
    taxonomy = get_taxonomy_info(species_name='Pongo abelii')
    print(taxonomy)
    
    # Test conservation status lookup
    print("\nLooking up Critically Endangered species:")
    cr_species = get_species_by_conservation_status('CR')
    print(cr_species[['sciName', 'common_name', 'conservation_status']])
