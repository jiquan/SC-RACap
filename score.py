import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from evaluate import load
from bert_score import BERTScorer
import warnings
from collections import defaultdict

from bert_score import BERTScorer, bert_cos_score_idf


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


class WeightedBERTScorer(BERTScorer):
    """
    A BERTScore Scorer that applies custom weights to specific vocabulary,
    similar to idf weighting.
    """

    def __init__(self, custom_vocab=None, custom_weight_factor=2.0, *args,
                 **kwargs):
        """
        Args:
            - custom_vocab (list of str): list of phrases for which to apply custom weights
            - custom_weight_factor (float): factor by which to increase weights for custom tokens
        """
        super().__init__(*args, **kwargs)
        self._custom_weight_dict = None  # Dictionary to hold custom weights
        self.custom_vocab = custom_vocab or []
        self.custom_weight_factor = custom_weight_factor
        if self.custom_vocab:
            self.compute_custom_weights()

    def compute_custom_weights(self):
        """
        Generate custom weights for the tokens in custom_vocab phrases.
        """
        if self._custom_weight_dict is not None:
            warnings.warn("Overwriting the previous custom weights.")

        # Initialize the custom weight dictionary with default weight of 1.0
        self._custom_weight_dict = defaultdict(lambda: 1.0)

        # Process each phrase in custom_vocab
        for phrase in self.custom_vocab:
            # Tokenize the phrase and get token IDs
            tokens = self._tokenizer.tokenize(phrase)
            token_ids = self._tokenizer.convert_tokens_to_ids(tokens)

            # Increase the weight for each token ID in this phrase
            for token_id in token_ids:
                # Multiply the existing weight by the custom factor
                self._custom_weight_dict[token_id] = self.custom_weight_factor

    def score(self, cands, refs, verbose=False, batch_size=64,
              return_hash=False):
        """
        Override score method to apply custom weights along with idf.
        """
        # Ensure that custom weights are computed
        if self._custom_weight_dict is None:
            raise ValueError(
                "Custom weights have not been computed. Please run `compute_custom_weights` first.")

        # Use `custom_weight_dict` in scoring
        all_preds = bert_cos_score_idf(
            self._model,
            refs,
            cands,
            self._tokenizer,
            self._custom_weight_dict,
            # Pass custom weights as the weight dictionary
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            all_layers=self.all_layers,
        ).cpu()

        # Apply baseline rescaling if necessary
        if self.rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (
                    1 - self.baseline_vals)

        # Return the same structure as in BERTScorer
        out = all_preds[..., 0], all_preds[..., 1], all_preds[
            ..., 2]  # P, R, F
        if return_hash:
            out = tuple([out, self.hash])

        return out

special_vocab = [
    'bipolar forceps',
    'large needle driver',
    'prograsp forceps',
    'monopolar curved scissors',
    'clip applier',
    'suction',
    'stapler',
    'ultrasound probe',
    'looping',
    'manipulating',
    'clipping',
    'cauterizing',
    'retracting',
    'grasping',
    'cutting',
    'suctioning',
    'suturing',
    'stapling',
    'sensing',
    'tissue',
    'blood',
    'vessels'
]

bert_scorer = WeightedBERTScorer(
    model_type='minilmv2-bertscore-distilled',
    num_layers=6,
    custom_vocab=special_vocab,
    custom_weight_factor=10,
)

def compute_bertscore(gt_list, pred_list):

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    # results = bertscore.compute(predictions=pred_list, references=gt_list,
    #                             lang="en")
    precision, recall, f1 = bert_scorer.score(pred_list, gt_list)
    # precision = sum(results['precision']) / len(results['precision'])
    # recall = sum(results['recall']) / len(results['recall'])
    # f1 = sum(results['f1']) / len(results['f1'])
    print('precision:{}'.format(precision.mean().item()))
    print('recall:{}'.format(recall.mean().item()))
    print('f1:{}'.format(f1.mean().item()))
    return precision.mean().item(), recall.mean().item(), f1.mean().item()


def evaluate_metrics(gt_dict, pred_dict):
    scorer = Scorer(pred_dict, gt_dict)
    total_scores = scorer.compute_scores()

    return total_scores


def evaluate():
    gt_path = './data/2018miccai/val.jsonl'
    pred_path = './generated_captions/best.json'
    scores_dict = dict()
    vid_gt = dict()
    vid_pred = dict()
    with open(gt_path, "r") as jsonl_file:
        for line in jsonl_file:
            try:
                # 解析JSON数据并将其添加到data列表中
                data_item = json.loads(line)
                if isinstance(data_item['text'], str):
                    # vid_gt[data_item['file_name']] = [data_item['text']]
                    vid_gt[data_item['file_name']] = [data_item['text'].lower()]
                elif isinstance(data_item['text'], list):
                    # vid_gt[data_item['file_name']] = data_item['text']
                    vid_gt[data_item['file_name']] = [item.lower() for item in
                                                      data_item['text']]
            except json.JSONDecodeError as e:
                # 处理JSON解析错误，如果发生错误的话
                print(f"Error parsing JSON: {str(e)}")
    with open(pred_path, 'r',
              encoding='utf-8') as file:
        pred_data = json.load(file)
        for data_item in pred_data:
            vid_pred[data_item['image_name']] = [data_item['prediction'].replace('\n', '')]
    total_scores = evaluate_metrics(vid_gt, vid_pred)
    gt_list = [lst[0] for lst in vid_gt.values()]
    pred_list = [lst[0] for lst in vid_pred.values()]

    print(total_scores)
    compute_bertscore(gt_list, pred_list)

    # scores_dict['all'] = total_scores
    # with open('eval.json', 'w') as file:
    #     json.dump(scores_dict, file, indent=4)


if __name__ == '__main__':
    evaluate()
    # compute_bertscore()