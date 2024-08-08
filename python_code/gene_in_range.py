
# import pyensembl
#
# # Initialize the Ensembl reference
# ensembl = pyensembl.EnsemblRelease(104)

import requests

def get_genes(chromosome, start, end):
    server = "https://rest.ensembl.org"
    ext = f"/overlap/region/human/{chromosome}:{start}-{end}?feature=gene"
    headers = {"Content-Type": "application/json"}

    response = requests.get(server + ext, headers=headers)
    if not response.ok:
        response.raise_for_status()

    return response.json()

data = [
    ("chr1-1245493-1248050", "SDF4"),
    ("chr1-1330394-1334148", "MRPL20"),
    ("chr1-2145904-2147150", "FAAP20"),
    ("chr1-9713011-9736481", "PIK3CD"),
    ("chr1-21287896-21301043", "ECE1"),
    ("chr1-23615606-23621287", "RPL11"),
    ("chr1-24500773-24509089", "NIPAL3"),
    ("chr1-24534067-24537250", "RCAN3"),
    ("chr1-24909406-24919504", "RUNX3"),
    ("chr1-24920221-24933567", "RUNX3"),
    ("chr1-27348860-27351636", "SYTL1"),
    ("chr1-27978911-27980217", "SMPDL3B"),
    ("chr1-32246575-32257651", "LCK"),
    ("chr1-39540336-39541916", "PABPC4"),
    ("chr1-40421192-40425654", "SMAP2"),
    ("chr1-44719250-44725697", "PLK3"),
    ("chr1-44817773-44821403", "PLK3"),
    ("chr1-45569650-45570662", "NASP"),
    ("chr1-84450811-84451342", "RPF1"),
    ("chr1-89259090-89260730", "GBP5")
]

parsed_data = []

for region in data:
    chromosome = region[0].split("chr")[1].split("-")[0]
    start = region[0].split("-")[1]
    end = region[0].split("-")[2]
    gene_name = region[1]

    # Get genes in the specified range
    genes_in_range = get_genes(chromosome, start, end)

    # Print the genes
    gene_in_range = False
    for gene in genes_in_range:
        try:
            # print(f"\tGene {gene['external_name']}")
            if gene['external_name'] == gene_name:
                gene_in_range = True
        except:
            print('No gene in this region')

    print(f'Chr: {chromosome}, start: {start}, end: {end}, gene_name: {gene_name}, gene_in_range: {gene_in_range}')