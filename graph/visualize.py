from graph.graph_builder import build_graph
import matplotlib.pyplot as plt

def visualize_graph():
    """
    Generate and save the graph visualization as pipeline_graph.png
    """
    try:
        graph = build_graph()
        app = graph.compile()

        # Generate the PNG using pygraphviz or similar
        # If this fails, we skip visualization but don't stop the pipeline
        png_data = app.get_graph().draw_png()
        with open("pipeline_graph.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as pipeline_graph.png")
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
        print("Tip: Install pygraphviz (requires graphviz-dev) or use mermaid output.")

if __name__ == "__main__":
    visualize_graph()
