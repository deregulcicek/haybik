import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_ssa_conceptual_diagram():
    """Create SSA conceptual diagram showing the four types and their relationships"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Synthetic Social Alienation (SSA) Conceptual Framework', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Central concept
    central_box = FancyBboxPatch((4, 4.5), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#ff6b6b', 
                                edgecolor='black', 
                                linewidth=2)
    ax.add_patch(central_box)
    ax.text(5, 5, 'Digital\nPlatforms', fontsize=12, fontweight='bold', 
            ha='center', va='center')
    
    # Type 1: Algorithmic Manipulation
    type1_box = FancyBboxPatch((0.5, 7), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#4ecdc4', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(type1_box)
    ax.text(1.75, 7.75, 'Type 1:\nAlgorithmic\nManipulation', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Type 2: Digital Alienation
    type2_box = FancyBboxPatch((7, 7), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#45b7d1', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(type2_box)
    ax.text(8.25, 7.75, 'Type 2:\nDigital\nAlienation', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Type 3: Platform Dependency
    type3_box = FancyBboxPatch((0.5, 1), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#96ceb4', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(type3_box)
    ax.text(1.75, 1.75, 'Type 3:\nPlatform\nDependency', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Type 4: Echo Chamber Effects
    type4_box = FancyBboxPatch((7, 1), 2.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#feca57', 
                               edgecolor='black', 
                               linewidth=2)
    ax.add_patch(type4_box)
    ax.text(8.25, 1.75, 'Type 4:\nEcho Chamber\nEffects', 
            fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Connection arrows
    # Type 1 to central
    arrow1 = ConnectionPatch((3, 7.75), (4, 5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow1)
    ax.text(3.5, 6.5, 'Content\nControl', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Type 2 to central
    arrow2 = ConnectionPatch((7, 7.75), (6, 5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow2)
    ax.text(6.5, 6.5, 'Social\nDisconnection', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Type 3 to central
    arrow3 = ConnectionPatch((3, 1.75), (4, 4.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow3)
    ax.text(3.5, 3.2, 'Behavioral\nReliance', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Type 4 to central
    arrow4 = ConnectionPatch((7, 1.75), (6, 4.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow4)
    ax.text(6.5, 3.2, 'Belief\nReinforcement', fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # User experience indicators
    ax.text(5, 3.8, 'User Experience', fontsize=11, fontweight='bold', 
            ha='center', va='center')
    
    # Examples for each type
    examples_y = 0.3
    ax.text(1.75, examples_y, 'Examples:\n• Algorithm control\n• Feed manipulation\n• Content filtering', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#4ecdc4', alpha=0.3))
    
    ax.text(8.25, examples_y, 'Examples:\n• Social isolation\n• Virtual vs real\n• Communication barriers', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#45b7d1', alpha=0.3))
    
    ax.text(1.75, examples_y + 2, 'Examples:\n• Platform addiction\n• Information dependency\n• Daily routine control', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#96ceb4', alpha=0.3))
    
    ax.text(8.25, examples_y + 2, 'Examples:\n• Filter bubbles\n• Confirmation bias\n• Opinion polarization', 
            fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#feca57', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('ssa_conceptual_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SSA conceptual diagram saved as 'ssa_conceptual_diagram.png'")

if __name__ == "__main__":
    create_ssa_conceptual_diagram() 