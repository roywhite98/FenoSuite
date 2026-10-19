"""Register the independent vector-root beta tab and background tools."""
from plant_phenotype.app.plugin_api import PluginManifest
from plant_phenotype.app.tab_manager import TabContribution
from .tools import RootVectorAnalyzeTool, RootVectorResolveTool


def get_plugin():
    return BuiltinRootPhenotypeBetaPlugin()


class BuiltinRootPhenotypeBetaPlugin:
    manifest = PluginManifest(
        id="builtin.root_phenotype_beta", name="根系表型beta", version="0.1.0",
        description="方向场辅助的矢量根系与父子关系解析",
    )

    def register(self, app):
        from .tab import build_root_phenotype_beta_tab
        app.register_tab(TabContribution(
            id="builtin.root_phenotype_beta.workspace", title="根系表型beta",
            factory=lambda app, parent=None: build_root_phenotype_beta_tab(app, parent), order=45,
        ))
        app.register_tool(RootVectorAnalyzeTool())
        app.register_tool(RootVectorResolveTool())
