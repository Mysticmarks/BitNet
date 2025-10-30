# Documentation Review Checklist

_This checklist is enforced by CI (`utils/check_docs_sync.py`). Complete it whenever interfaces change._

- [ ] Regenerate the architecture diagrams: `python utils/generate_architecture_docs.py`
- [ ] Refresh the API reference: `python utils/generate_api_reference.py`
- [ ] Rebuild the changelog: `python utils/generate_changelog.py`
- [ ] Confirm README and SRD mention any new limitations, flags, or configuration requirements.
- [ ] Update the roadmap with milestone owners or dates if timelines shift.
- [ ] Add or adjust deployment guidance when runtime flags or outputs change.
- [ ] Note test coverage additions or gaps in `docs/iteration-log.md` when relevant.
