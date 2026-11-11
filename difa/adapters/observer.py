"""Read SiT intermediate tensors during inference without modifying models.py.

The hooks observe the same before/middle/after tensors as the measured G03
callback implementation. All handles are removed after each forward, including
on exceptions. No tensor output is replaced; no model parameter is added.
"""


class SiTObserverAdapter:
    def __init__(self, model):
        self.wrapped = model

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, x, t, y, structure_observer=None):
        if structure_observer is None:
            return self.wrapped(x, t, y)
        if self.wrapped.training:
            raise ValueError('DiFA structural observations require model.eval()')
        handles = []
        readout = {}

        def install_block(index, block):
            recorded = {}

            def before_block(module, args):
                recorded['before'], recorded['condition'] = args[0], args[1]

            def before_norm2(module, args):
                recorded['middle'] = args[0]

            def after_block(module, args, output):
                structure_observer.block(index, recorded['before'],
                                         recorded['middle'], output,
                                         recorded['condition'])

            handles.append(block.register_forward_pre_hook(before_block))
            handles.append(block.norm2.register_forward_pre_hook(before_norm2))
            handles.append(block.register_forward_hook(after_block))

        def before_head(module, args):
            readout['features'], readout['condition'] = args[0], args[1]

        try:
            for index, block in enumerate(self.wrapped.blocks):
                if structure_observer.wants(index, len(self.wrapped.blocks)):
                    install_block(index, block)
            handles.append(self.wrapped.final_layer.register_forward_pre_hook(before_head))
            output = self.wrapped(x, t, y)
            structure_observer.finish(readout['features'], readout['condition'], output)
            return output
        finally:
            for handle in handles:
                handle.remove()
