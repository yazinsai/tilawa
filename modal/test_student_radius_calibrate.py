import unittest
from student_radius_calibrate import calibrate


class RadiusCalibrationTest(unittest.TestCase):
    def parts(self):
        rows=[]
        for speaker, positive, energy in [("a",True,.5),("b",True,.7),("a",False,.1),("c",False,.3)]:
            rows.append(dict(speaker=speaker,split="train",positive=positive,
                result=dict(status="crossed",energy=energy),primarySha256="sha",inventory="phones",
                search=dict(dimensions=4,iterations=8,difference=.05,maxEnergy=4)))
        return [dict(schema=1,shard=0,shards=1,excludedSpeakers=[],recordings=4,rows=rows)]

    def test_speaker_validation_freezes_preservation_cut(self):
        model,report=calibrate(self.parts())
        self.assertEqual(model["rejectionEnergy"],.5)
        self.assertEqual(report["frozenFalseVetoed"],2)
        self.assertEqual(report["frozenTrueLost"],0)

    def test_excluded_speakers_and_missing_shards_fail_closed(self):
        parts=self.parts();parts[0]["excludedSpeakers"]=["a"]
        with self.assertRaises(ValueError): calibrate(parts)
        parts=self.parts();parts[0]["shards"]=2
        with self.assertRaises(ValueError): calibrate(parts)

    def test_a_failed_search_cannot_create_a_calibration_cut(self):
        parts=self.parts()
        for row in parts[0]["rows"]:
            row["result"]=dict(status="no-crossing",energy=None)
        with self.assertRaises(ValueError): calibrate(parts)


if __name__=="__main__": unittest.main()
