from tests.testing_harness import TestHarness


def test_trigger_no_status(request):
    harness = TestHarness('statepoint.10.h5')
    harness.request = request
    harness.main()
