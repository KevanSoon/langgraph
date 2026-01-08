import React from "react";
import { Layout, Pointer, Zap } from "lucide-react";

import { Feature108 } from "./ui/shadcnblocks-com-feature108"

const demoData = {
  badge: "Platform Features",
  heading: "Everything you need to build faster",
  description: "Comprehensive tools and features designed to help you scale your application effortlessly.",
  tabs: [
    {
      value: "tab-1",
      icon: <Zap className="h-auto w-4 shrink-0" />,
      label: "Performance",
      content: {
        badge: "High Efficiency",
        title: "Maximize your revenue potential.",
        description:
          "Optimize your workflow with high-performance tools that ensure your application runs smoothly and efficiently under any load.",
        buttonText: "Learn More",
        imageSrc:
          "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2670&auto=format&fit=crop",
        imageAlt: "Data analysis chart",
      },
    },
    {
      value: "tab-2",
      icon: <Pointer className="h-auto w-4 shrink-0" />,
      label: "Engagement",
      content: {
        badge: "User Focused",
        title: "Engage your users effectively.",
        description:
          "Leverage powerful engagement metrics and tools to understand user behavior and keep them coming back for more.",
        buttonText: "See Tools",
        imageSrc:
          "https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=2670&auto=format&fit=crop",
        imageAlt: "Team collaboration",
      },
    },
    {
      value: "tab-3",
      icon: <Layout className="h-auto w-4 shrink-0" />,
      label: "Design",
      content: {
        badge: "Pixel Perfect",
        title: "Craft stunning digital experiences.",
        description:
          "Access a library of beautiful, pre-built layouts that you can customize to match your brand identity perfectly.",
        buttonText: "Explore Designs",
        imageSrc:
          "https://images.unsplash.com/photo-1467232004584-a241de8bcf5d?q=80&w=2669&auto=format&fit=crop",
        imageAlt: "Web design workspace",
      },
    },
  ],
};

function Feature108Demo() {
  return <Feature108 {...demoData} />;
}

export { Feature108Demo };