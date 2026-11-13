import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location("gpu_group_check",Path(__file__).with_name("gpu-group-check.py"))
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class CleanupTests(unittest.TestCase):
    def test_inference_failure_already_released_by_node_is_not_cleanup_failure(self):
        unload=Mock()
        self.assertEqual(check.cleanup_owned_instance(lambda:None,unload,"old"),"already_released")
        unload.assert_not_called()

    def test_replacement_instance_is_not_unloaded_by_old_test(self):
        unload=Mock()
        self.assertEqual(check.cleanup_owned_instance(lambda:{"instance":{"instance_id":"new"}},unload,"old"),"replacement_preserved")
        unload.assert_not_called()

    def test_own_instance_is_unloaded(self):
        unload=Mock()
        self.assertEqual(check.cleanup_owned_instance(lambda:{"instance":{"instance_id":"old"}},unload,"old"),"unloaded")
        unload.assert_called_once_with()


if __name__=="__main__":
    unittest.main()
