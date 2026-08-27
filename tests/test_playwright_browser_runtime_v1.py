import unittest

from twelve_six.playwright_browser import (
    BROWSER_REVISION,
    ContractError,
    GREENLET_VERSION,
    PACKAGE_VERSION,
    PYYYEE_VERSION,
    TYPING_EXTENSIONS_VERSION,
    RuntimeContract,
    UPSTREAM_COMMIT,
    validate_action,
    validate_selector,
    validate_url,
)


class PlaywrightBrowserRuntimeTests(unittest.TestCase):
    def test_dependency_closure_is_pinned(self):
        self.assertEqual(PACKAGE_VERSION, "1.62.0")
        self.assertEqual(PYYYEE_VERSION, "13.0.1")
        self.assertEqual(GREENLET_VERSION, "3.2.3")
        self.assertEqual(TYPING_EXTENSIONS_VERSION, "4.16.0")

    def test_identity_is_immutable_and_pinned(self):
        self.assertEqual(UPSTREAM_COMMIT, "e3950d9c140d007bd52853b45813c6274b24e36f")
        self.assertEqual(BROWSER_REVISION, "1234")

    def test_allowed_semantic_action(self):
        validate_action("click_element")

    def test_forbidden_arbitrary_action(self):
        with self.assertRaises(ContractError):
            validate_action("evaluate_javascript")

    def test_network_requires_explicit_allowlist(self):
        with self.assertRaises(ContractError):
            validate_url("https://example.com", network_mode="deny")
        validate_url(
            "https://example.com/path",
            network_mode="allowlist",
            allowed_hosts=frozenset({"example.com"}),
        )

    def test_network_allowlist_is_fail_closed(self):
        with self.assertRaises(ContractError):
            validate_url(
                "https://example.com",
                network_mode="allowlist",
                allowed_hosts=frozenset(),
            )

    def test_local_data_page_is_allowed_without_network(self):
        validate_url("data:text/html,<p>ok</p>", network_mode="deny")
        validate_url("about:blank", network_mode="deny")

    def test_coordinate_or_xpath_selector_is_rejected(self):
        with self.assertRaises(ContractError):
            validate_selector("coordinates:12,44")
        with self.assertRaises(ContractError):
            validate_selector("xpath=//button")

    def test_credential_environment_is_rejected(self):
        contract = RuntimeContract(env_allowlist=("GITHUB_TOKEN",))
        with self.assertRaises(ContractError):
            contract.validate()

    def test_unknown_network_mode_is_rejected(self):
        with self.assertRaises(ContractError):
            RuntimeContract(network_mode="anywhere").validate()

    def test_exact_runtime_drift_is_rejected(self):
        import twelve_six.playwright_browser as module
        original = module.metadata.version
        module.metadata.version = lambda _: "1.57.0"
        try:
            with self.assertRaisesRegex(ContractError, "1.62.0 required"):
                module.installed_identity()
        finally:
            module.metadata.version = original

    def test_runtime_absence_is_rejected(self):
        import twelve_six.playwright_browser as module
        original = module.metadata.version
        module.metadata.version = (
            lambda _: (_ for _ in ()).throw(module.metadata.PackageNotFoundError)
        )
        try:
            with self.assertRaisesRegex(ContractError, "not installed"):
                module.installed_identity()
        finally:
            module.metadata.version = original


if __name__ == "__main__":
    unittest.main()
