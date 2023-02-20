import argparse

from datasets import load_dataset

from apo.scorers import PIIScorer

scorer = PIIScorer()


def process_doc(doc):
    sents = doc['texts']
    scores = [scorer.score_text(sent)/len(sent) for sent in sents]
    return {
        'texts': sents,
        'meta': doc['meta'],
        'scores': scores,
        'avg_score': sum(scores) / len(scores),
        'num_sents': len(doc),
    }


def score_pii(start_id: int, end_id: int, num_proc: int):
    dataset = load_dataset(f"tomekkorbak/detoxify-pile-chunk3-{start_id}-{end_id}", split='train')
    dataset = dataset.map(process_doc, batch_size=16, num_proc=num_proc)
    dataset.push_to_hub(f"tomekkorbak/pii-pile-chunk3-{start_id}-{end_id}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, required=True, default=0)
    parser.add_argument('--end_id', type=int, required=True, default=50_000)
    parser.add_argument('--num_proc', type=int, required=True, default=16)
    args = parser.parse_args()
    score_pii(start_id=args.start_id, end_id=args.end_id, num_proc=args.num_proc)
