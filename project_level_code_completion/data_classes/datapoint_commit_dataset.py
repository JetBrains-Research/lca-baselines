from data_classes.datapoint_base import DatapointBase


class DatapointCommitDataset(DatapointBase):
    def _get_py_context(self) -> dict[str, str]:
        context = self.context_dict
        py_context = {fn: cont for fn, cont in context.items() if fn.endswith('.py')}
        return py_context

    def _get_non_py_context(self) -> dict[str, str]:
        context = self.context_dict
        non_py_context = {fn: cont for fn, cont in context.items() if not fn.endswith('.py')}
        return non_py_context

    def get_context(self) -> dict[str, str]:
        py_context = self._get_py_context()
        non_py_context = self._get_non_py_context()
        non_py_keys = list(non_py_context)
        non_py_idx = 0
        context = py_context.copy()
        context_len = sum([len(code) for code in context.values()])
        while non_py_idx < len(non_py_keys):
            curr_fn = non_py_keys[non_py_idx]
            curr_cont = non_py_context[curr_fn]
            if not (curr_fn.startswith('.') or 'license' in curr_fn.lower()):
                if len(curr_cont) < 16000:
                    context[curr_fn] = curr_cont
            non_py_idx += 1
            context_len = sum([len(code) for code in context.values()])

        return context
