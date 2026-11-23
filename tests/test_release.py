import json
from pathlib import Path
import tempfile
import unittest

import torch

from fcr_boc.models.gated_lora import GatedLoRALinear
from release_utils import metrics, read_rows, FIELDS


class ReleaseTests(unittest.TestCase):
    def test_average_precision_groups_ties(self):
        self.assertAlmostEqual(metrics([0,1,0,1],[0.,0.,0.,0.])['ap'],50.)
        self.assertAlmostEqual(metrics([0,0,1,1],[-2.,-1.,1.,2.])['ap'],100.)
        self.assertAlmostEqual(metrics([0,1,1,0],[3.,2.,2.,1.])['ap'],200/3)

    def test_rejects_other_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'stylegan'/'image.png';image.parent.mkdir();image.touch()
            row=dict(image_path=str(image),label='real',**{k:'Evidence.' for k in FIELDS})
            manifest=root/'rows.jsonl';manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Only GAN-Test/ProGAN'):
                read_rows(manifest,progan_only=True)

    def test_zero_initialized_adapters_preserve_base(self):
        torch.manual_seed(7)
        base=torch.nn.Linear(8,8)
        layer=GatedLoRALinear(base,rank=4,alpha=16.,dropout=0.,
                             fcr_enable=True,corrector_rank=2,boc_enable=True)
        layer.eval();x=torch.randn(2,5,8)
        self.assertTrue(torch.equal(layer(x),base(x)))
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))


if __name__=='__main__':
    unittest.main()
