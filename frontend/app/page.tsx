"use client"
import { AcmeHero } from '@/components/ui/acme-hero';
import { ThemeToggle } from '../components/ThemeToggle';
import AnoAI from '@/components/ui/animated-shader-background';
import { GlowingEffectDemo } from '@/components/glowing-effect-demo';
import { Feature108Demo } from '@/components/feature108-demo';
import { TestimonialsDemo } from '@/components/testimonials-demo';

export default function App() {
  return (
    <div className="relative min-h-screen font-sans antialiased text-foreground">

      {/* Background Shader */}
      <div className="fixed inset-0 z-[-1] ">
        <AnoAI />
      </div>
      {/* Main Content */}
      <div className="relative z-10 flex flex-col gap-16 pb-20">
        <AcmeHero />
        
        <section className="container max-w-6xl mx-auto px-4">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-bold tracking-tight mb-4">Features</h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Discover what makes our platform stand out with these powerful capabilities designed for modern development.
            </p>
          </div>
          <GlowingEffectDemo />
        </section>

        <section className="container max-w-7xl mx-auto px-4">
           <Feature108Demo />
        </section>

        <section className="container max-w-7xl mx-auto px-4">
           <TestimonialsDemo />
        </section>
      </div>
    </div>
  );
}