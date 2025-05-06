from manim import *
import numpy as np

class ReseauxDeNeurones(Scene):
    def construct(self):
        # Couleurs et styles
        INPUT_COLOR = "#2196F3"  # Bleu
        HIDDEN_COLOR = "#9C27B0"  # Violet
        OUTPUT_COLOR = "#E91E63"  # Rose
        NEURON_RADIUS = 0.15
        CONNECTION_OPACITY = 0.7
        BG_COLOR = "#1A1A1A"  # Fond noir légèrement plus clair
        
        self.camera.background_color = BG_COLOR
        
        # ========== INTRO (0:00-0:05) ==========
        titre = Text("Les Réseaux de Neurones", font="Arimo", color=WHITE).scale(1.2)
        sous_titre = Text("En 60 secondes ⏱️", font="Arimo", color="#FFC107").scale(0.8)
        sous_titre.next_to(titre, DOWN)
        
        intro_group = VGroup(titre, sous_titre)
        
        # Logo d'intro (simplifié - un cerveau stylisé)
        # cerveau = SVGMobject("https://raw.githubusercontent.com/3b1b/manim/master/logo/brain.svg")
        # cerveau.set_fill(opacity=0.8)
        # cerveau.set_color(BLUE_C)
        # cerveau.scale(0.5)
        # cerveau.next_to(intro_group, UP, buff=0.5)
        
        # intro_all = VGroup(cerveau, intro_group)
        
        # # Animation d'intro
        # self.play(FadeIn(cerveau, shift=UP), run_time=0.5)
        # self.play(Write(titre), run_time=1)
        # self.play(FadeIn(sous_titre, shift=UP*0.5), run_time=0.5)
        # self.wait(0.5)
        # self.play(FadeOut(intro_all), run_time=0.5)
        
        # ========== SEGMENT 1 (0:05-0:25) - Structure d'un réseau ==========
        # Création du réseau de neurones stylisé
        def create_network():
            layers = [3, 4, 4, 2]  # Structure: entrée, cachées, sortie
            network = VGroup()
            neurons = []
            
            # Créer les couches de neurones
            x_spacing = 3
            max_neurons = max(layers)
            
            # Pour chaque couche
            for i, layer_size in enumerate(layers):
                layer = VGroup()
                layer_neurons = []
                
                # Position verticale pour centrer la couche
                y_offset = (max_neurons - layer_size) / 2
                
                for j in range(layer_size):
                    # Choisir la couleur selon le type de couche
                    if i == 0:
                        color = INPUT_COLOR
                    elif i == len(layers) - 1:
                        color = OUTPUT_COLOR
                    else:
                        color = HIDDEN_COLOR
                    
                    neuron = Circle(radius=NEURON_RADIUS, fill_opacity=1, color=color)
                    neuron.move_to([i * x_spacing - (len(layers)-1) * x_spacing / 2, 
                                   j - layer_size/2 + y_offset, 0])
                    
                    layer.add(neuron)
                    layer_neurons.append(neuron)
                
                neurons.append(layer_neurons)
                network.add(layer)
            
            # Ajouter les connexions entre neurones
            connections = VGroup()
            
            for i in range(len(layers) - 1):
                for neuron1 in neurons[i]:
                    for neuron2 in neurons[i+1]:
                        connection = Line(
                            neuron1.get_center(), 
                            neuron2.get_center(),
                            stroke_opacity=CONNECTION_OPACITY,
                            stroke_width=0.7
                        )
                        connections.add(connection)
            
            return VGroup(connections, network), neurons
        
        network_group, neurons = create_network()
        
        # Labels pour les différentes couches
        input_label = Text("Couche d'entrée", font="Arimo", color=INPUT_COLOR).scale(0.5)
        hidden_label1 = Text("Couches cachées", font="Arimo", color=HIDDEN_COLOR).scale(0.5)
        output_label = Text("Couche de sortie", font="Arimo", color=OUTPUT_COLOR).scale(0.5)
        
        # Positionnement des labels
        input_label.next_to(VGroup(*neurons[0]), DOWN, buff=0.5)
        hidden_label1.next_to(VGroup(*neurons[1], *neurons[2]), DOWN, buff=0.5)
        output_label.next_to(VGroup(*neurons[3]), DOWN, buff=0.5)
        
        labels = VGroup(input_label, hidden_label1, output_label)
        
        # Animation du segment 1
        self.play(
            Write(Text("Un réseau de neurones, c'est comme un cerveau miniature.", font="Arimo").scale(0.7).to_edge(UP)),
            run_time=1.5
        )
        
        # Montrer le réseau progressivement
        self.play(
            FadeIn(network_group[1]),  # Neurones d'abord
            run_time=1
        )
        self.play(
            Create(network_group[0]),  # Puis les connexions
            run_time=1.5
        )
        
        self.play(
            Write(labels),
            run_time=1.5
        )
        
        self.wait(1)
        self.clear()
        
        # ========== SEGMENT 2 (0:25-0:45) - Les poids et l'apprentissage ==========
        # Visualisation simplifiée de poids
        small_network = VGroup()
        
        # Créer un petit réseau à 2 couches pour démontrer les poids
        layer1 = VGroup(*[Circle(radius=NEURON_RADIUS, fill_opacity=1, color=INPUT_COLOR) 
                         for _ in range(2)])
        layer1.arrange(DOWN, buff=1)
        layer1.to_edge(LEFT, buff=3)
        
        layer2 = VGroup(*[Circle(radius=NEURON_RADIUS, fill_opacity=1, color=OUTPUT_COLOR) 
                         for _ in range(2)])
        layer2.arrange(DOWN, buff=1)
        layer2.to_edge(RIGHT, buff=3)
        
        small_network.add(layer1, layer2)
        
        # Créer les connexions avec des poids
        connections = VGroup()
        weights_values = [[0.2, 0.8], [0.5, 0.3]]  # Valeurs d'exemple
        weight_labels = VGroup()
        
        for i, neuron1 in enumerate(layer1):
            for j, neuron2 in enumerate(layer2):
                connection = Line(
                    neuron1.get_center(),
                    neuron2.get_center(),
                    stroke_opacity=0.8
                )
                # Fléchage directionnel
                arrow_tip = ArrowTip(scale=0.1)
                arrow_tip.set_color(WHITE)
                connection.add_tip(arrow_tip)
                
                # Label de poids
                weight = weights_values[i][j]
                weight_label = Text(f"{weight}", font="Arimo").scale(0.4)
                weight_label.set_color(YELLOW)
                weight_label.move_to(connection.get_center() + UP*0.2)
                
                connections.add(connection)
                weight_labels.add(weight_label)
        
        # Titre pour cette section
        titre_poids = Text("Les poids des connexions", font="Arimo").scale(0.8)
        titre_poids.to_edge(UP)
        
        # Afficher le petit réseau et ses poids
        self.play(
            FadeIn(small_network),
            Write(titre_poids),
            run_time=1
        )
        self.play(
            Create(connections),
            run_time=1
        )
        
        self.play(
            Write(weight_labels),
            run_time=1
        )
        
        # Animation montrant l'apprentissage (changement des poids)
        new_weights = [[0.4, 0.6], [0.7, 0.2]]
        new_weight_labels = VGroup()
        
        for i, (neuron1, conn_pair) in enumerate(zip(layer1, zip(weights_values, new_weights))):
            for j, (neuron2, (old_w, new_w)) in enumerate(zip(layer2, zip(conn_pair[0], conn_pair[1]))):
                new_label = Text(f"{new_w}", font="Arimo").scale(0.4)
                new_label.set_color(GREEN)
                new_label.move_to(weight_labels[i*len(layer2) + j].get_center())
                new_weight_labels.add(new_label)
        
        apprentissage_text = Text("Apprentissage: ajustement des poids", font="Arimo", color=GREEN).scale(0.6)
        apprentissage_text.to_edge(DOWN, buff=1)
        
        self.play(
            Write(apprentissage_text),
            run_time=0.8
        )
        
        self.play(
            *[Transform(old_label, new_label) for old_label, new_label in zip(weight_labels, new_weight_labels)],
            run_time=2
        )
        
        self.wait(1)
        self.clear()
        
        # ========== SEGMENT 3 (0:45-1:00) - Applications ==========
        
        # Titre de la section
        titre_applications = Text("Applications des réseaux de neurones", font="Arimo").scale(0.8)
        titre_applications.to_edge(UP)
        
        # Créer des icônes pour différentes applications
        app_icons = VGroup()
        
        # Icône pour la reconnaissance faciale
        face_recognition = Circle(radius=0.6, color=BLUE)
        face_features = VGroup(
            Circle(radius=0.1, color=WHITE).move_to([-0.2, 0.1, 0]), # Œil gauche
            Circle(radius=0.1, color=WHITE).move_to([0.2, 0.1, 0]),  # Œil droit
            Arc(angle=-PI/2, radius=0.3, color=WHITE).rotate(PI).shift(DOWN*0.1) # Sourire
        )
        scanning_line = Line([-0.6, 0.6, 0], [0.6, 0.6, 0], color=GREEN)
        face_recognition_group = VGroup(face_recognition, face_features, scanning_line)
        
        # Icône pour la traduction automatique
        text_translation = Rectangle(width=1.2, height=0.8, color=ORANGE)
        text1 = Text("Bonjour", font="Arimo").scale(0.3)
        text2 = Text("Hello", font="Arimo").scale(0.3)
        arrow = Arrow(LEFT, RIGHT, color=WHITE).scale(0.5)
        text1.next_to(arrow, LEFT)
        text2.next_to(arrow, RIGHT)
        translation_group = VGroup(text_translation, text1, arrow, text2)
        
        # Icône pour la prédiction médicale
        medical = Circle(radius=0.6, color=RED)
        cross = VGroup(
            Line(UP*0.4, DOWN*0.4, color=WHITE),
            Line(LEFT*0.4, RIGHT*0.4, color=WHITE)
        )
        heartbeat = VGroup()
        points = [[-0.4, 0, 0], [-0.2, 0, 0], [-0.1, 0.3, 0], [0, 0, 0], [0.1, -0.3, 0], [0.2, 0, 0], [0.4, 0, 0]]
        for i in range(len(points)-1):
            heartbeat.add(Line(points[i], points[i+1], color=WHITE))
        medical_group = VGroup(medical, cross, heartbeat)
        
        # Organisation des icônes
        face_recognition_group.scale(0.8)
        translation_group.scale(0.8)
        medical_group.scale(0.8)
        
        app_icons.add(face_recognition_group, translation_group, medical_group)
        app_icons.arrange(RIGHT, buff=1)
        app_icons.next_to(titre_applications, DOWN, buff=0.8)
        
        # Ajouter les labels sous chaque icône
        labels = VGroup(
            Text("Reconnaissance faciale", font="Arimo").scale(0.4),
            Text("Traduction", font="Arimo").scale(0.4),
            Text("Diagnostic médical", font="Arimo").scale(0.4)
        )
        
        for label, icon in zip(labels, app_icons):
            label.next_to(icon, DOWN, buff=0.3)
        
        # Animation finale
        self.play(
            Write(titre_applications),
            run_time=1
        )
        
        self.play(
            *[FadeIn(icon) for icon in app_icons],
            run_time=1.5
        )
        
        self.play(
            Write(labels),
            run_time=1
        )
        
        # Animation de la ligne de scan sur le visage
        self.play(
            scanning_line.animate.shift(DOWN*1.2),
            run_time=1,
            rate_func=linear
        )
        
        # Animation conclusion
        conclusion = Text("C'est ça la magie de l'apprentissage automatique!", 
                          font="Arimo", color=YELLOW).scale(0.6)
        conclusion.to_edge(DOWN, buff=0.5)
        
        self.play(
            Write(conclusion),
            run_time=1
        )
        
        self.wait(1)
        
        # ========== OUTRO ==========
        outro = VGroup(
            Text("Abonne-toi pour d'autres explications!", font="Arimo").scale(0.7),
            Text("sur l'IA et les technologies du futur 🚀", font="Arimo").scale(0.6)
        )
        outro.arrange(DOWN)
        
        # Animation de transition vers l'outro
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=0.5
        )
        
        self.play(
            Write(outro),
            run_time=1
        )