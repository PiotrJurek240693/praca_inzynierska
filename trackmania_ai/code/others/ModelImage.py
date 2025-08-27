from code.ppo.PPO import Actor, Critic
from torchview import draw_graph

if __name__ == "__main__":
    actor = Actor(38, 2).eval()
    graph = draw_graph(actor.body, input_size=(1, 38), expand_nested=True)
    graph.visual_graph.render("../../images/actor_model", format="png")

    critic = Critic(38).eval()
    graph = draw_graph(critic, input_size=(1, 38), expand_nested=True)
    graph.visual_graph.render("../../images/critic_model", format="png")
