import os
from typing import Iterable, Tuple, Any
from itertools import islice
from time import perf_counter
import argparse

import spacy
from detoxify import Detoxify
from datasets import load_dataset, Dataset
from tqdm import tqdm

spacy_model = spacy.blank("en")
sentencizer = spacy_model.add_pipe("sentencizer")
spacy_model.max_length = 1e12
detoxify_model = Detoxify('original', device='cuda')


def get_raw_text_and_meta(documents: Iterable[dict[str, Any]]) -> Iterable[Tuple[str]]:
    for document in documents:
        yield document['text'], document['meta']


def split_sentences(documents: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    raw_texts = get_raw_text_and_meta(documents)
    for idx, (spacy_doc, meta) in enumerate(spacy_model.pipe(raw_texts, n_process=8, as_tuples=True)):
        for sent in spacy_doc.sents:
            yield {'text': sent.text_with_ws, 'meta': meta, 'idx': idx}


def classify(sents: Iterable[dict[str, Any]], batch_size: int = 1024) -> Iterable[dict[str, Any]]:
    sents = iter(sents)
    while True:
        batch = list(islice(sents, batch_size))
        if len(batch) > 0:
            raw_texts = [sent['text'] for sent in batch]
            for score, sent in zip(detoxify_model.predict(raw_texts)['toxicity'], batch):
                yield {'score': score, **sent}
        else:
            break


def construct_doc(doc: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'texts': [sent['text'] for sent in doc],
        'meta': doc[0]['meta'],
        'scores': [sent['score'] for sent in doc],
        'avg_score': sum(sent['score'] for sent in doc) / len(doc),
        'num_sents': len(doc),
    }


def join_sentences(sents: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    prev_idx = -1
    current_doc = []
    for sent in sents:
        if sent['idx'] == prev_idx:
            current_doc.append(sent)
        else:
            if prev_idx != -1:
                yield construct_doc(current_doc)
            current_doc = [sent]
            prev_idx = sent['idx']
    yield construct_doc(current_doc)


def get_documents(
        dataset: Iterable[dict[str, Any]],
        start_idx: int,
        stop_idx: int,
        detoxify_batch_size: int
) -> Iterable[dict[str, Any]]:
    total_docs =  stop_idx - start_idx
    yield from tqdm(join_sentences(
        classify(
            sents=split_sentences(
                islice(dataset, start_idx, stop_idx)
            ),
            batch_size=detoxify_batch_size)
    ), total=total_docs)


def test_pipeline() -> None:
    num_docs = 10
    pile1 = load_dataset('the_pile', streaming=True, split='train')
    pile2 = load_dataset('the_pile', streaming=True, split='train')
    it1, it2 = get_documents(pile1, 0, num_docs), get_documents(pile2, 0, num_docs)
    for i, (processed_doc, original_doc) in enumerate(zip(it1, it2)):
        assert ''.join(processed_doc['texts']) == original_doc['text']
        assert processed_doc['meta'] == original_doc['meta']
        assert len(processed_doc['scores']) == processed_doc['num_sents']

    assert i == num_docs-1


def score(
        start_idx: int, stop_idx: int,
        pile_chunk_idx: int,
        output_dataset_name: str,
        detoxify_batch_size: int
) -> None:
    print(f'Scoring {stop_idx-start_idx} documents from pile chunk {pile_chunk_idx} starting at index {start_idx}')
    start_time = perf_counter()
    pile_chunk = load_dataset(
        "/scratch/work/public/ml-datasets/pile/train",
        data_files={'train': f'{pile_chunk_idx:02d}.jsonl'},
        split='train',
        streaming=True
    )
    new_dataset = Dataset.from_generator(
        get_documents,
        gen_kwargs={
            'dataset': pile_chunk,
            'start_idx': start_idx,
            'stop_idx': stop_idx,
            'detoxify_batch_size': detoxify_batch_size
        },
    )
    print(f'Finished scoring in {perf_counter() - start_time:.2f}s')
    new_dataset.push_to_hub(output_dataset_name, token=os.environ['HUGGING_FACE_HUB_TOKEN'])
    print(f'Time elapsed: {(perf_counter() - start_time):.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--stop_idx', type=int, default=10_000)
    parser.add_argument('--output_dataset_name', type=str)
    parser.add_argument('--pile_chunk_idx', type=int, default=0)
    parser.add_argument('--detoxify_batch_size', type=int, default=512)
    args = parser.parse_args()
    score(
        start_idx=args.start_idx,
        stop_idx=args.stop_idx,
        pile_chunk_idx=args.pile_chunk_idx,
        output_dataset_name=args.output_dataset_name,
        detoxify_batch_size=args.detoxify_batch_size
    )
