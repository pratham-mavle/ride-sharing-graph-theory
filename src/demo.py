import networkx as nx
import matplotlib.pyplot as plt

def create_demo_graph():
    G = nx.Graph()

    drivers = ["D1", "D2", "D3"]
    passengers = ["P1", "P2", "P3"]

    G.add_nodes_from(drivers, bipartite=0)
    G.add_nodes_from(passengers, bipartite=1)

    # edges with score
    edges = [
        ("D1", "P1", 10),
        ("D2", "P1", 8),
        ("D3", "P1", 12),
        ("D1", "P2", 9),
        ("D2", "P2", 7),
        ("D3", "P2", 11),
        ("D1", "P3", 14),
        ("D2", "P3", 6),
        ("D3", "P3", 10),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G


def draw_graph(G, title):
    pos = nx.spring_layout(G)

    edge_labels = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()


def simulate_matching(G):
    print("\n--- Matching Process ---")

    matches = []

    for passenger in ["P1", "P2", "P3"]:
        edges = G.edges(passenger, data=True)

        best_edge = min(edges, key=lambda x: x[2]['weight'])

        driver = best_edge[0]

        print(f"{passenger} matched with {driver} (score={best_edge[2]['weight']})")

        matches.append((driver, passenger))

        # remove driver so it cannot be reused
        G.remove_node(driver)

    return matches


def main():
    G = create_demo_graph()

    draw_graph(G, "Initial Graph")

    matches = simulate_matching(G)

    print("\nFinal Matches:", matches)


if __name__ == "__main__":
    main()