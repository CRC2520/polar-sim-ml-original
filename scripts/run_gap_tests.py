"""Record actual per-test success, not a claimed acceptance count."""
from pathlib import Path
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.passed=[]
    def addSuccess(self,test):super().addSuccess(test);self.passed.append(test.id())
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
payload={'run':result.testsRun,'passed_count':len(result.passed),'passed':result.passed,
         'failures':[str(x[0]) for x in result.failures],'errors':[str(x[0]) for x in result.errors],
         'status':'passed' if result.wasSuccessful() else 'failed'}
(ROOT/'docs/P0P2_TEST_REPORT.json').write_text(json.dumps(payload,indent=2)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
