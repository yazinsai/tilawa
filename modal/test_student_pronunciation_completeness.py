"""Training construction checks: removed source content cannot leak into negatives."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('completeness',Path(__file__).with_name('student_pronunciation_completeness.py'))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CompletenessTests(unittest.TestCase):
    def test_contiguous_crops_and_pronunciation_balance(self):
        row=dict(speaker='training-fixture',features=[[float(i)]*32 for i in range(30)],
          words=[{"from":6,"to":18,"ids":[1,2,3],"phoneBounds":[[6,10],[10,14],[14,18]]}])
        data=m.examples(row)
        self.assertEqual(len(data),8)
        self.assertEqual(sum(e['y'] for e in data),4)
        for e in data:
            source=[frame[0] for frame in e['x']]
            self.assertEqual(source,list(range(int(source[0]),int(source[-1])+1)))
            self.assertEqual(e['ids'],[1,2,3])
        negatives=[e for e in data if not e['y']]
        self.assertEqual([len(e['x']) for e in negatives],[8,12,8,12])
        self.assertEqual(negatives[1]['x'][0][0],10)
        self.assertEqual(negatives[3]['x'][-1][0],13)

    def test_uncertain_views_include_original_without_crossing_window(self):
        views=m.crop_views(list(range(5)))
        self.assertEqual(views,[[0,1,2,3,4],[1,2,3,4],[0,1,2,3]])
        self.assertEqual(m.crop_views([1]),[[1],[1],[1]])

if __name__=='__main__': unittest.main()
