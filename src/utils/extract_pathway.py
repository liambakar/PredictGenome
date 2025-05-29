import pandas as pd
import numpy as np

def load_hallmark_signatures(path):
    """
    Returns dict[pathway] = list of gene names
    """
    df = pd.read_csv(path)
    return {col: df[col].dropna().unique().tolist() for col in df.columns}

def extract_pathway_summary(expr_df, signatures):
    """
    For each sample and each pathway, perform:
      1) Extract subvector according to pathway gene list
      2) R(x ⊙ a) —— remove all 0 elements
    Returns:
      sample_ids: list of sample identifiers (first column of expr_df)
      pathway_summary: dict[pathway] = list of np.ndarray,
                       pathway_summary[c][i] is the R aggregated vector for sample i on pathway c
    """
    # Assume first column of expr_df is sample ID, remaining columns are gene expression
    sample_ids = expr_df.iloc[:, 0].tolist()
    gene_columns = expr_df.columns[1:]
    data = expr_df[gene_columns].values  # shape = (num_samples, N_g)

    # Pre-compute column indices for each pathway in the gene matrix
    pathway_idx = {}
    for pathway, genes in signatures.items():
        # Filter genes that exist in expr_df and convert to column indices (starting from 0 corresponding to column 0 in data)
        idx = [gene_columns.get_loc(g) for g in genes if g in gene_columns]
        pathway_idx[pathway] = idx

    # Aggregation: for each pathway, iterate through all samples
    pathway_summary = {c: [] for c in signatures}
    for i in range(data.shape[0]):            # For each sample
        x = data[i]                          # x is a vector of length Ng
        for c, idx in pathway_idx.items():   # For each pathway
            if len(idx) > 0:
                # 1) Extract subvector  2) Remove zero values
                vals = x[idx]
                nonzero = vals[vals != 0]
                pathway_summary[c].append(nonzero)
            else:
                # If this pathway has no genes in this dataset, set empty vector
                pathway_summary[c].append(np.array([], dtype=x.dtype))
    return sample_ids, pathway_summary

if __name__ == '__main__':
    rna_path = '../datasets/rna_clean.csv'

    Hallmark_pathway_path = '../datasets/hallmarks_signatures.csv'
    pd_hallmark = pd.read_csv(Hallmark_pathway_path)

    expr = pd.read_csv(rna_path)  
    signatures = load_hallmark_signatures(Hallmark_pathway_path)
    sample_ids, S_path = extract_pathway_summary(expr, signatures)

    print("=== S_path Shape Analysis ===")
    print(f"Number of pathways: {len(S_path)}")
    print(f"Number of samples: {len(sample_ids)}")
    
    print(f"\n=== Pathway-Sample Consistency Check ===")
    pathway_sample_counts = {}
    for pathway, vectors in S_path.items():
        pathway_sample_counts[pathway] = len(vectors)
    
    unique_counts = set(pathway_sample_counts.values())
    print(f"Unique sample counts across pathways: {unique_counts}")
    
    if len(unique_counts) == 1:
        print(f"✅ All pathways have exactly {list(unique_counts)[0]} samples")
    else:
        print(f"❌ Inconsistent sample counts across pathways!")
        for pathway, count in list(pathway_sample_counts.items())[:10]:  # Show first 10
            print(f"  {pathway}: {count} samples")
    
    print(f"\nFirst sample ID: {sample_ids[0]}")
    print("Available pathways (first 5):", list(S_path.keys())[:5])
    
    print("\n=== Pathway Vector Lengths (for first sample) ===")
    for i, (pathway, vectors) in enumerate(S_path.items()):
        if i < 5:  #
            vector_length = len(vectors[0]) if len(vectors) > 0 else 0
            print(f"{pathway}: {vector_length} non-zero genes")
    
    # if "HALLMARK_ANDROGEN_RESPONSE" in S_path:
    #     androgen_vectors = S_path["HALLMARK_ANDROGEN_RESPONSE"]
    #     print(f"\n=== HALLMARK_ANDROGEN_RESPONSE Analysis ===")
    #     print(f"Number of samples: {len(androgen_vectors)}")
    #     print(f"Vector lengths across samples: {[len(v) for v in androgen_vectors[:5]]}...")  # 前5个样本的向量长度
        
    #     lengths = [len(v) for v in androgen_vectors]
    #     print(f"Min vector length: {min(lengths)}")
    #     print(f"Max vector length: {max(lengths)}")
    #     print(f"Mean vector length: {np.mean(lengths):.2f}")
    
    print(f"\n=== Overall Statistics ===")
    total_vectors = sum(len(vectors) for vectors in S_path.values())
    print(f"Total pathway-sample combinations: {total_vectors}")
    print(f"Expected total (pathways × samples): {len(S_path)} × {len(sample_ids)} = {len(S_path) * len(sample_ids)}")
    
    if total_vectors == len(S_path) * len(sample_ids):
        print("✅ Data structure is consistent!")
    else:
        print("❌ Data structure inconsistency detected!")
