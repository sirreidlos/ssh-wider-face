{
  description = "Dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          cacert
        ];

        shellHook = ''
          echo "Activating Python virtualenv..."

          if [ -d .venv ]; then
            exec fish -c "source .venv/bin/activate.fish; exec fish"
          else
            exec fish
          fi
        '';
      };
    };
}
