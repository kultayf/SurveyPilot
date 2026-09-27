"""命令行评测入口：validate、score、compare，以及可选的 Ragas 模型评测。"""
import argparse
import asyncio
import json
from pathlib import Path
from .benchmark import read_cases, score, compare, fingerprint


def main():
    parser = argparse.ArgumentParser(description='SurveyPilot 可复现科研评测')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate'); validate.add_argument('cases')
    scoring = commands.add_parser('score'); scoring.add_argument('cases'); scoring.add_argument('run')
    scoring.add_argument('--annotations'); scoring.add_argument('--k', type=int, default=5); scoring.add_argument('--output', required=True)
    paired = commands.add_parser('compare'); paired.add_argument('before'); paired.add_argument('after'); paired.add_argument('--output', required=True)
    running = commands.add_parser('run'); running.add_argument('cases'); running.add_argument('--profile', required=True)
    running.add_argument('--limit', type=int, default=1); running.add_argument('--storage-root', required=True); running.add_argument('--output', required=True)
    qa = commands.add_parser('corpus-run'); qa.add_argument('cases'); qa.add_argument('--corpus', required=True)
    qa.add_argument('--mode', choices=['bm25','hybrid'], default='bm25'); qa.add_argument('--limit', type=int, default=1); qa.add_argument('--output', required=True)
    retrieval = commands.add_parser('corpus-retrieve'); retrieval.add_argument('cases'); retrieval.add_argument('--corpus', required=True)
    retrieval.add_argument('--limit', type=int, default=1); retrieval.add_argument('--output', required=True)
    ragas = commands.add_parser('ragas'); ragas.add_argument('cases'); ragas.add_argument('run')
    ragas.add_argument('--model', required=True); ragas.add_argument('--embedding-model'); ragas.add_argument('--faithfulness-only', action='store_true'); ragas.add_argument('--output', required=True)
    ragas.add_argument('--max-tokens', type=int, default=8192); ragas.add_argument('--reasoning-effort')
    packet = commands.add_parser('review-packet'); packet.add_argument('cases'); packet.add_argument('run')
    packet.add_argument('--packet', required=True); packet.add_argument('--annotations-output', required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    try:
        if args.command == 'validate':
            cases = read_cases(args.cases)
            print(json.dumps({'cases': len(cases), 'verified': sum(c.get('annotation_status') == 'verified' for c in cases),
                              'dataset_hash': fingerprint(cases)}, ensure_ascii=False, indent=2))
            return
        if args.command == 'corpus-retrieve':
            from .corpus_runner import retrieve_corpus
            result = asyncio.run(retrieve_corpus(read_cases(args.cases), args.corpus, args.output, args.limit))
        elif args.command == 'corpus-run':
            from .corpus_runner import run_corpus
            result = asyncio.run(run_corpus(read_cases(args.cases), args.corpus, args.output, args.mode, args.limit))
        elif args.command == 'run':
            from .workflow_runner import run_cases
            result = asyncio.run(run_cases(read_cases(args.cases), load(args.profile), args.output, args.storage_root, args.limit))
        elif args.command == 'review-packet':
            from .review_packet import prepare_review_packet
            result = prepare_review_packet(read_cases(args.cases), load(args.run), args.packet, args.annotations_output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        elif args.command == 'score':
            result = score(read_cases(args.cases), load(args.run), args.k, load(args.annotations) if args.annotations else None)
        elif args.command == 'compare':
            result = compare(load(args.before), load(args.after))
        else:
            from .ragas_runner import evaluate_run
            result = evaluate_run(read_cases(args.cases), load(args.run), args.model, args.embedding_model,
                                  faithfulness_only=args.faithfulness_only, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f'已保存：{path.resolve()}')
    except (ValueError, KeyError, TypeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
