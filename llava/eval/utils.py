from .bert_score import BERTScorer

class BertEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']

        gts = [] # ground truth
        res = [] # prediction
        for imgId in imgIds:
            gts_per = []
            for c in self.coco.imgToAnns[imgId]:
                gts_per.append(c['caption'])
            gts.append(gts_per)
            res.append(self.cocoRes.imgToAnns[imgId][0]['caption'])
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        P, R, F1 = scorer.score(res, gts)
        
        self.eval['bert-f1'] = F1.mean().item()
        self.eval['bert-re'] = R.mean().item()
        self.eval['bert-pr'] = P.mean().item()

        for k, v in self.eval.items():
            print('%s: %.3f'%(k, v))