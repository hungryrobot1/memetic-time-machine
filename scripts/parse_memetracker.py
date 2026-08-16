#!/usr/bin/env python3
"""
Parse MemeTracker cluster data and extract curated meme samples.

Usage:
    python parse_memetracker.py clust-qt08080902w3mfq5.txt.gz --output ../data/memes_curated.json

Data format (from SNAP):
    A:  <ClSz>  <TotFq>  <Root>  <ClId>
    B:          <QtFq>   <Urls>  <QtStr>  <QtId>
    C:                   <Tm>    <Fq>     <UrlTy>  <Url>
"""

import gzip
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Iterator, Dict, List, Any


def parse_clusters(filepath: str, max_clusters: int = None) -> Iterator[Dict]:
    """
    Parse MemeTracker cluster file, yielding one cluster dict at a time.
    """
    open_fn = gzip.open if filepath.endswith('.gz') else open
    
    current_cluster = None
    current_phrase = None
    cluster_count = 0
    
    with open_fn(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            
            parts = line.split('\t')
            
            # Detect record type by structure
            # A records: ClSz, TotFq, Root, ClId (4 fields, first is small int)
            # B records: QtFq, Urls, QtStr, QtId (4 fields, starts with freq)
            # C records: Tm, Fq, UrlTy, Url (4 fields, starts with timestamp)
            
            if len(parts) >= 4:
                # Try to detect A record (new cluster)
                try:
                    cl_sz = int(parts[0])
                    tot_fq = int(parts[1])
                    root = parts[2]
                    cl_id = int(parts[3])
                    
                    # This looks like an A record
                    if current_cluster is not None:
                        yield current_cluster
                        cluster_count += 1
                        if max_clusters and cluster_count >= max_clusters:
                            return
                    
                    current_cluster = {
                        'cluster_id': cl_id,
                        'cluster_size': cl_sz,
                        'total_frequency': tot_fq,
                        'root_phrase': root,
                        'phrases': []
                    }
                    current_phrase = None
                    continue
                except (ValueError, IndexError):
                    pass
                
                # Try to detect B record (phrase within cluster)
                try:
                    qt_fq = int(parts[0])
                    urls = int(parts[1])
                    qt_str = parts[2]
                    qt_id = int(parts[3])
                    
                    current_phrase = {
                        'phrase_id': qt_id,
                        'phrase': qt_str,
                        'frequency': qt_fq,
                        'url_count': urls,
                        'mentions': []
                    }
                    if current_cluster is not None:
                        current_cluster['phrases'].append(current_phrase)
                    continue
                except (ValueError, IndexError):
                    pass
                
                # Try to detect C record (URL mention)
                if current_phrase is not None and len(parts) >= 4:
                    try:
                        # Timestamp format: 2008-08-18 14:23:05
                        tm = parts[0]
                        fq = int(parts[1])
                        url_ty = parts[2]
                        url = parts[3]
                        
                        current_phrase['mentions'].append({
                            'timestamp': tm,
                            'frequency': fq,
                            'source_type': 'blog' if url_ty == 'B' else 'mainstream',
                            'url': url
                        })
                    except (ValueError, IndexError):
                        pass
    
    # Yield final cluster
    if current_cluster is not None:
        yield current_cluster


def compute_cluster_stats(cluster: Dict) -> Dict:
    """Compute temporal and structural statistics for a cluster."""
    all_timestamps = []
    blog_count = 0
    mainstream_count = 0
    
    for phrase in cluster['phrases']:
        for mention in phrase['mentions']:
            try:
                ts = datetime.strptime(mention['timestamp'], '%Y-%m-%d %H:%M:%S')
                all_timestamps.append(ts)
                if mention['source_type'] == 'blog':
                    blog_count += 1
                else:
                    mainstream_count += 1
            except:
                pass
    
    if all_timestamps:
        all_timestamps.sort()
        median_idx = len(all_timestamps) // 2
        peak_time = all_timestamps[median_idx]
        
        return {
            'first_mention': all_timestamps[0].isoformat(),
            'peak_time': peak_time.isoformat(),
            'last_mention': all_timestamps[-1].isoformat(),
            'duration_hours': (all_timestamps[-1] - all_timestamps[0]).total_seconds() / 3600,
            'blog_mentions': blog_count,
            'mainstream_mentions': mainstream_count,
            'blog_fraction': blog_count / (blog_count + mainstream_count) if (blog_count + mainstream_count) > 0 else 0
        }
    return {}


def extract_top_clusters(filepath: str, n: int = 100) -> List[Dict]:
    """Extract the top N clusters by total frequency."""
    clusters = []
    
    print(f"Parsing clusters from {filepath}...")
    for i, cluster in enumerate(parse_clusters(filepath)):
        clusters.append(cluster)
        if (i + 1) % 1000 == 0:
            print(f"  Parsed {i + 1} clusters...")
    
    print(f"Total clusters parsed: {len(clusters)}")
    
    # Sort by total frequency
    clusters.sort(key=lambda c: c['total_frequency'], reverse=True)
    
    # Take top N and compute stats
    top_clusters = clusters[:n]
    for cluster in top_clusters:
        cluster['stats'] = compute_cluster_stats(cluster)
    
    return top_clusters


def curate_for_experiment(clusters: List[Dict], 
                          strong_count: int = 5,
                          weak_count: int = 5) -> Dict:
    """
    Curate a dataset for the activation geometry experiment.
    
    Strong memes: Top clusters by frequency (viral, many variants)
    Weak memes: Mid-tier clusters with similar root phrase length but lower spread
    """
    # Strong memes: top N by frequency
    strong = clusters[:strong_count]
    
    # Weak memes: clusters ranked around 500-1000 with decent variant count
    # These propagated but didn't achieve viral status
    weak_candidates = [c for c in clusters[400:1000] if c['cluster_size'] >= 3]
    weak = weak_candidates[:weak_count] if len(weak_candidates) >= weak_count else weak_candidates
    
    return {
        'metadata': {
            'source': 'MemeTracker (SNAP Stanford)',
            'paper': 'Leskovec, Backstrom, Kleinberg. Meme-tracking and the Dynamics of the News Cycle. KDD 2009.',
            'date_range': 'August 2008 - October 2008',
            'selection_criteria': {
                'strong': f'Top {strong_count} clusters by total frequency',
                'weak': f'{weak_count} clusters from rank 400-1000 with 3+ variants'
            }
        },
        'strong_memes': [
            {
                'cluster_id': c['cluster_id'],
                'root_phrase': c['root_phrase'],
                'total_frequency': c['total_frequency'],
                'variant_count': c['cluster_size'],
                'variants': [p['phrase'] for p in c['phrases'][:10]],  # Top 10 variants
                'stats': c.get('stats', {})
            }
            for c in strong
        ],
        'weak_memes': [
            {
                'cluster_id': c['cluster_id'],
                'root_phrase': c['root_phrase'],
                'total_frequency': c['total_frequency'],
                'variant_count': c['cluster_size'],
                'variants': [p['phrase'] for p in c['phrases'][:10]],
                'stats': c.get('stats', {})
            }
            for c in weak
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Parse MemeTracker cluster data')
    parser.add_argument('input', help='Path to clust-qt08080902w3mfq5.txt.gz')
    parser.add_argument('--output', '-o', default='memes_curated.json',
                        help='Output JSON file')
    parser.add_argument('--top-n', type=int, default=1000,
                        help='Number of top clusters to analyze')
    parser.add_argument('--strong', type=int, default=5,
                        help='Number of strong memes to select')
    parser.add_argument('--weak', type=int, default=5,
                        help='Number of weak memes to select')
    
    args = parser.parse_args()
    
    clusters = extract_top_clusters(args.input, n=args.top_n)
    curated = curate_for_experiment(clusters, args.strong, args.weak)
    
    with open(args.output, 'w') as f:
        json.dump(curated, f, indent=2)
    
    print(f"\nCurated dataset written to {args.output}")
    print(f"  Strong memes: {len(curated['strong_memes'])}")
    print(f"  Weak memes: {len(curated['weak_memes'])}")
    
    print("\nStrong meme root phrases:")
    for m in curated['strong_memes']:
        print(f"  - {m['root_phrase'][:60]}... (freq: {m['total_frequency']})")


if __name__ == '__main__':
    main()
